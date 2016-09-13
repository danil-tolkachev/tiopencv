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

__attribute__((reqd_work_group_size(1,1,1))) __kernel void tidsp_gaussian(__global const uchar* srcptr, int srcStep, int srcOffset,
                                                                          __global uchar*dstptr, int dstStep, int dstOffset, int rows, int cols)
{
  uchar *y_start_ptr = (uchar *)srcptr;
  uchar * restrict dest_ptr;
  uchar * restrict y_ptr[LINES_CACHED];
  uchar result;
  int   rd_idx, start_rd_idx, fetch_rd_idx;
  int   gid   = get_global_id(0);
  EdmaMgr_Handle evIN  = EdmaMgr_alloc(LINES_CACHED);
  unsigned int srcPtr[LINES_CACHED], dstPtr[LINES_CACHED], numBytes[LINES_CACHED];
  local uchar img_lines[LINES_CACHED+1][MAX_LINE_SIZE]; // LINES_CACHED lines needed for processing and one more inflight via EDMA
  int clk_start, clk_end;
  int i, j, kk;
  long r0, r1, r2;
  unsigned int out0, out1, out2, out3, out10, out32, r0_5432, r1_5432, r2_5432;
  unsigned int mask_r0_0 = 0x00010201U;
  unsigned int mask_r0_1 = mask_r0_0 << 8U;
  unsigned int mask_r1_0 = 0x00020402U;
  unsigned int mask_r1_1 = mask_r1_0 << 8U;

  if (!evIN) { printf("Failed to alloc edmaIN handle.\n"); return; }
#ifdef TIDSP_OPENCL_VERBOSE
  clk_start = __clock();
#endif
  rows >>= 1;
  dest_ptr = (uchar *)dstptr;

  if(gid == 0)
  { /* Upper half of image */
    memset (img_lines, 0, (LINES_CACHED - 1) * MAX_LINE_SIZE);
    EdmaMgr_copy1D1D(evIN, (void*)(srcptr), (void*)img_lines[LINES_CACHED - 1], cols);
    fetch_rd_idx = cols;
  } else if(gid == 1)
  { /* Bottom half of image */
    for(i = LINES_CACHED - 1; i >= 0; i --) {
      srcPtr[i] = (unsigned int)(srcptr + (rows - i) * cols);
      dstPtr[i] = (unsigned int)img_lines[i];  
      numBytes[i] = cols;
    }
    EdmaMgr_copy1D1DLinked(evIN, srcPtr, dstPtr, numBytes, LINES_CACHED);
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
    EdmaMgr_copyFast(evIN, (void*)(srcptr + fetch_rd_idx), (void*)(&img_lines[rd_idx]));
    fetch_rd_idx += cols;
    /* Case 1: SIMD of 4 output pixels at a time */
    /* 6/4 = 1.5 cycles per pixel */
    for(i = 0; i < (cols/4); i++)
    {
        /* Read 8 bytes from each of the 3 lines.  Only need 6 bytes,
         * so there is potential for reading 2 bytes beyond input buffer
         * in some cases, but these 2 bytes are not used  by the algo.  */
        r0 = _mem8_const(&(y_ptr[0])[i*4]);
        r1 = _mem8_const(&(y_ptr[1])[i*4]);
        r2 = _mem8_const(&(y_ptr[2])[i*4]);

        /* Output 0: Convolve the 9 neighborhood pixels */
        out0 = (_dotpu4(_loll(r0), mask_r0_0) +
                          _dotpu4(_loll(r1), mask_r1_0) +
                          _dotpu4(_loll(r2), mask_r0_0)  ) / 16U;

        /* Output 1: Convolve the 9 neighborhood pixels */
        out1 = (_dotpu4(_loll(r0), mask_r0_1) +
                          _dotpu4(_loll(r1), mask_r1_1) +
                          _dotpu4(_loll(r2), mask_r0_1)  ) / 16U;

        out10 = _pack2(out1, out0);

        r0_5432 = _packlh2(_hill(r0), _loll(r0));
        r1_5432 = _packlh2(_hill(r1), _loll(r1));
        r2_5432 = _packlh2(_hill(r2), _loll(r2));

        /* Output 2: Convolve the 9 neighborhood pixels */
        out2 = (_dotpu4(r0_5432, mask_r0_0) +
                          _dotpu4(r1_5432, mask_r1_0) +
                          _dotpu4(r2_5432, mask_r0_0)  ) / 16U;

        /* Output 3: Convolve the 9 neighborhood pixels */
        out3 = (_dotpu4(r0_5432, mask_r0_1) +
                          _dotpu4(r1_5432, mask_r1_1) +
                          _dotpu4(r2_5432, mask_r0_1)  ) / 16U;

        out32 = _pack2(out3, out2);

        _mem4(&dest_ptr[i*4]) = _spacku4((int)out32, (int)out10);
    }
    dest_ptr += cols;
  }
  EdmaMgr_wait(evIN);
  EdmaMgr_free(evIN);

#ifdef TIDSP_OPENCL_VERBOSE
  clk_end = __clock();
  printf ("TIDSP gauss clockdiff=%d\n", clk_end - clk_start);
#endif
}
/********************************************************************************************/
