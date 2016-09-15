//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Zero Lin, zero.lin@amd.com
//    Yao Wang, bitwangyaoyao@gmail.com
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//

#include <dsp_c.h>
#include <edmamgr.h>

#define MAX_LINE_SIZE 2048
#define LINES_CACHED  3

__attribute__((reqd_work_group_size(1,1,1))) __kernel void tidsp_sobel(__global const uchar* srcptr, int srcStep, int srcOffset,
                                                                       __global uchar*dstptr, int dstStep, int dstOffset, int rows, int cols)
{
  uchar *y_start_ptr = (uchar *)srcptr;
  short * restrict dest_ptr;
  uchar * restrict y_ptr[LINES_CACHED];
  uchar *yprev_ptr, *ycurr_ptr, *ynext_ptr;
  uchar result;
  int   rd_idx, start_rd_idx, fetch_rd_idx;
  int   gid   = get_global_id(0);
  EdmaMgr_Handle evIN  = EdmaMgr_alloc(LINES_CACHED);
  local uchar img_lines[LINES_CACHED+1][MAX_LINE_SIZE]; // LINES_CACHED lines needed for processing and one more inflight via EDMA
  int clk_start, clk_end;
  int i, j, kk;

#ifdef SOBELCOEF
  long mask1_8 = 0x0101010101010101;
  long mask2_8 = 0x0202020202020202;
#else
  long mask1_8 = 0x0303030303030303;
  long mask2_8 = 0x0a0a0a0a0a0a0a0a;
#endif
  unsigned int mask1_4 = _loll(mask1_8);
  unsigned int mask2_4 = _loll(mask2_8);
  long    r0, r1, r2;
  ushort8 r0_2, r1_2, r2_2;

  /* Used for x output */
  long r01_lo, r01_hi, r012_lo, r012_hi, r012_lo_d2, r012_hi_d2, c02_lo, c02_hi;

  /* Used for y output */
  long r02_lo, r02_hi, r02_lo_d1, r02_hi_d1, r02_lo_d2, r02_hi_d2, c01_lo, c01_hi, c012_lo, c012_hi;

  if (!evIN) { printf("Failed to alloc edmaIN handle.\n"); return; }

#ifdef TIDSP_OPENCL_VERBOSE
  clk_start = __clock();
#endif

  rows >>= 1;
  dest_ptr = (short *)dstptr;

  for(i = 0; i < (LINES_CACHED + 1); i ++)
  {
    memset ((void *)img_lines[i], 0, MAX_LINE_SIZE);
  }
  if(gid == 0)
  { /* Upper half of image */
    for(i = 1; i < LINES_CACHED; i ++)
    { /* Use this, one time multiple 1D1D transfers, instead of one linked transfer, to allow for fast EDMA later */
      EdmaMgr_copy1D1D(evIN, (void *)(srcptr + (rows - 1 + i) * cols), (void *)(img_lines[i]), cols);
    }
    fetch_rd_idx = cols;
  } else if(gid == 1)
  { /* Bottom half of image */
    for(i = 0; i < LINES_CACHED; i ++)
    { /* Use this, one time multiple 1D1D transfers, instead of one linked transfer, to allow for fast EDMA later */
      EdmaMgr_copy1D1D(evIN, (void *)(srcptr + (rows - 1 + i) * cols), (void *)(img_lines[i]), cols);
    }
    fetch_rd_idx = (rows + 1) * cols;
    dest_ptr += rows * cols;
  } else return;
  start_rd_idx = 0;

  for (int y = 0; y < rows; y ++)
  {
    EdmaMgr_wait(evIN);
    rd_idx  = start_rd_idx;
    for(kk = 0; kk < LINES_CACHED; kk ++)
    {
      y_ptr[kk] = (uchar *)img_lines[rd_idx];
      rd_idx = (rd_idx + 1) & LINES_CACHED;
    }
    start_rd_idx = (start_rd_idx + 1) & LINES_CACHED;
    EdmaMgr_copyFast(evIN, (void*)(srcptr + fetch_rd_idx), (void*)(img_lines[rd_idx]));
    fetch_rd_idx += cols;
    /**********************************************************************************/
    yprev_ptr = y_ptr[0];
    ycurr_ptr = y_ptr[1];
    ynext_ptr = y_ptr[2];

    /* Since unaligned loads are bottleneck, save cost of data fetch
     * by fetching first 2 columns and adding/subtracting cols.  Update this state
     * for every iteration. */
    unsigned int x0_3210 = _mem4_const(&yprev_ptr[0]);
    unsigned int x1_3210 = _mem4_const(&ycurr_ptr[0]);
    unsigned int x2_3210 = _mem4_const(&ynext_ptr[0]);

#ifdef X_OUTPUT
    /* Convert from 8bpp to 16bpp so we can do SIMD ops on rows */
    long x0_2 = _mpyu4ll(as_uint(x0_3210), as_uint(mask1_4));
    long x1_2 = _mpyu4ll(as_uint(x1_3210), as_uint(mask2_4));
    long x2_2 = _mpyu4ll(as_uint(x2_3210), as_uint(mask1_4));

    int x01            = _add2((unsigned int)_loll(x0_2), (unsigned int)_loll(x1_2));
    unsigned int xaxb  = (unsigned int)_add2(x01, (unsigned int)_loll(x2_2));
#endif

#ifdef Y_OUTPUT
    /* Convert from 8bpp to 16bpp so we can do SIMD subtraction of rows */
    long x0_2 = _mpyu4ll(as_uint(x0_3210), as_uint(mask1_4));
    long x2_2 = _mpyu4ll(as_uint(x2_3210), as_uint(mask1_4));

    /* Subtract row 0 from 2, column-wise */
    unsigned int yayb  = (unsigned int)_sub2((unsigned int)_loll(x2_2), (unsigned int)_loll(x0_2));
#endif
    /* Case 1: SIMD of 8 output pixels at a time */
    /* 7/8 = 0.875 cycles per pixel */
    /* NOTE: The case for aligned output was tried, but it could only achieve
     * better perfomance (13/16) if the loop was unrolled 2x, and it could not
     * be splooped, so I deemed not worth adding code, but can be added back if
     * needed */

    for(i = 0; i < (cols/8); i++)
    {
#ifdef X_OUTPUT
        /* Read 8 bytes from each of the 3 lines. */
        r0 = _mem8_const(&yprev_ptr[i*8+2]);
        r1 = _mem8_const(&ycurr_ptr[i*8+2]);
        r2 = _mem8_const(&ynext_ptr[i*8+2]);

        /* Convert from 8bpp to 16bpp so we can do SIMD of rows */
        r0_2 = _dmpyu4(as_uchar8(r0), as_uchar8(mask1_8));
        r1_2 = _dmpyu4(as_uchar8(r1), as_uchar8(mask2_8));
        r2_2 = _dmpyu4(as_uchar8(r2), as_uchar8(mask1_8));

            /* Add rows 0+1, column-wise */
            r01_lo = _dadd2(as_long(r0_2.s0123), as_long(r1_2.s0123));
            r01_hi = _dadd2(as_long(r0_2.s4567), as_long(r1_2.s4567));

            /* Add previous sum to row 2, column-wise */
            r012_lo = _dadd2(r01_lo, as_long(r2_2.s0123));
            r012_hi = _dadd2(r01_hi, as_long(r2_2.s4567));

            /* Left shift this sum by 2 pixels, using history */
            r012_lo_d2 = _itoll(_loll(r012_lo), xaxb);
            r012_hi_d2 = _itoll(_loll(r012_hi), _hill(r012_lo));

            /* Now subtract right from left columns to find filter result */
            /*    7 6 5 4 3 2 1 0 = r012
             *  - 5 4 3 2 1 0 a b = r012_d2
             * --------------------
             *              c02
             */
            c02_lo = _dsub2(r012_lo, r012_lo_d2);
            c02_hi = _dsub2(r012_hi, r012_hi_d2);

            /* save overlap data for next iteration */
            xaxb = (uint32_t)_mvd((int32_t)_hill(r012_hi));
        /* store 8 sobel x filter values */
        _mem8(&dest_ptr[i*8])     = c02_lo;
        _mem8(&dest_ptr[(i*8)+4]) = c02_hi;
#endif

#ifdef Y_OUTPUT
        /* Subtract row 0 from 2, column-wise (for y output) */
        r02_lo = _dsub2(as_long(r2_2.s0123), as_long(r0_2.s0123));
        r02_hi = _dsub2(as_long(r2_2.s4567), as_long(r0_2.s4567));

        /* Left shift this sum by one pixel, and 2 pixels respectively, using history */
        r02_lo_d1 = _itoll(_packlh2(_hill(r02_lo), _loll(r02_lo)),
                           _packlh2(_loll(r02_lo), yayb));
        r02_hi_d1 = _itoll(_packlh2(_hill(r02_hi), _loll(r02_hi)),
                           _packlh2(_loll(r02_hi), _hill(r02_lo)));

        r02_lo_d2 = _itoll(_loll(r02_lo), yayb);
        r02_hi_d2 = _itoll(_loll(r02_hi), _hill(r02_lo));

        /* Now add adjacent columns together to find neigborhood sums */
        /*    7 6 5 4 3 2 1 0 = r02
         *  + 6 5 4 3 2 1 0 a = r02_d1 * 2
         *  + 5 4 3 2 1 0 a b = r02_d2
         * --------------------
         *              c012
         */
        c01_lo  = _dadd2(r02_lo, _dshl2(r02_lo_d1, 1U));
        c012_lo = _dadd2(c01_lo, r02_lo_d2);

        c01_hi  = _dadd2(r02_hi, _dshl2(r02_hi_d1, 1U));
        c012_hi = _dadd2(c01_hi, r02_hi_d2);

        /* save overlap data for next iteration */
        yayb = (unsigned int)_mvd((int)_hill(r02_hi));

        /* store 8 sobel y filter values */
        _mem8(&dest_ptr[i*8])     = c012_lo;
        _mem8(&dest_ptr[(i*8)+4]) = c012_hi;
#endif
    }
    dest_ptr += cols;
  }
  EdmaMgr_wait(evIN);
  EdmaMgr_free(evIN);
#ifdef TIDSP_OPENCL_VERBOSE
  clk_end = __clock();
#ifdef X_OUTPUT
  printf ("TIDSP sobelX clockdiff=%d\n", clk_end - clk_start);
#endif
#ifdef Y_OUTPUT
  printf ("TIDSP sobelY clockdiff=%d\n", clk_end - clk_start);
#endif
#endif
}
/********************************************************************************************/
