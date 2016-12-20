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
#include <dsp_edmamgr.h>

#define MAX_LINE_SIZE 2048
#define LINES_CACHED  3
//#define TIDSP_OPENCL_VERBOSE


__attribute__((reqd_work_group_size(1,1,1))) __kernel void tidsp_minmaxValue(__global const uchar* srcptr, int srcStep, int srcOffset,
                                                                             int rows, int cols, __global uchar* dstptr)
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
  
  uint32_t min1, min2, min4;
  uint32_t max1, max2, max4;

  uint8_t *pMinVal = (uint8_t *)(dstptr + gid * 2 + 0);
  uint8_t *pMaxVal = (uint8_t *)(dstptr + gid * 2 + 1);
  /****************************/
  /* KERNEL SPECIFIC PROLOGUE */
  /****************************/
//  uint32_t  min = *pMinVal;
//  uint32_t  max = *pMaxVal;
  uint32_t min = 255;
  uint32_t max = 0;
  int64_t min8 = _itoll((min << 24) | (min << 16) | (min << 8) | min,
                        (min << 24) | (min << 16) | (min << 8) | min);
  int64_t max8 = _itoll((max << 24) | (max << 16) | (max << 8) | max,
                        (max << 24) | (max << 16) | (max << 8) | max);
  /****************************/

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
    /*******************/
    /* KERNEL SPECIFIC */
    /*******************/
    for(i=0; i < (cols / 8); i++) {
        int64_t src8 = _mem8_const(&ycurr_ptr[i*8]);
        min8 = _dminu4(src8, min8);
        max8 = _dmaxu4(src8, max8);
    }
    for(i*=8; i < cols; i++) {
        uint64_t min8_tmp = 0xFFFFFFFFFFFFFF00U | ycurr_ptr[i];
        int64_t  max8_tmp = (int64_t)ycurr_ptr[i];
        min8 = _dminu4((int64_t)min8_tmp, min8);
        max8 = _dmaxu4(max8_tmp, max8);
    }
    /*******************/
  }
  /****************************/
  /* KERNEL SPECIFIC EPILOGUE */
  /****************************/
  /* Reduce 8 lanes down to a single min and max value */
  min4 = _minu4(_hill(min8), _loll(min8));
  min2 = _minu4(min4, min4 >> 16);
  min1 = _minu4(min2, min2 >> 8) & 0xffU;

  max4 = _maxu4(_hill(max8), _loll(max8));
  max2 = _maxu4(max4, max4 >> 16);
  max1 = _maxu4(max2, max2 >> 8) & 0xffU;

  *pMinVal = (uint8_t)min1;
  *pMaxVal = (uint8_t)max1;
  /****************************/

  EdmaMgr_wait(evIN);
  EdmaMgr_free(evIN);
#ifdef TIDSP_OPENCL_VERBOSE
  clk_end = __clock();
  printf ("TIDSP minmax clockdiff=%d\n", clk_end - clk_start);
#endif
}
/********************************************************************************************/
