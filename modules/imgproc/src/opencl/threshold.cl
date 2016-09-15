#ifdef TIDSP_OPENCL
//-----------------------------
// Texas Instruments Inc., 2016
//-----------------------------
#include <dsp_c.h>
#include <edmamgr.h>

#define MAX_LINE_SIZE 2048
#define LINES_CACHED  3

__attribute__((reqd_work_group_size(1,1,1))) __kernel void tidsp_threshold(__global const uchar * srcptr, int src_step, int src_offset,
                        __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols, uchar thresh, uchar max_val, uchar min_val)

{
  uchar *y_start_ptr = (uchar *)srcptr;
  uchar * restrict dest_ptr;
  uchar * restrict y_ptr[LINES_CACHED];
  uchar *yprev_ptr, *ycurr_ptr, *ynext_ptr;
  int   rd_idx, start_rd_idx, fetch_rd_idx;
  int   gid   = get_global_id(0);
  EdmaMgr_Handle evIN  = EdmaMgr_alloc(LINES_CACHED);
  local uchar img_lines[LINES_CACHED+1][MAX_LINE_SIZE]; // LINES_CACHED lines needed for processing and one more inflight via EDMA
  int clk_start, clk_end;
  int i, j, kk;

  /*** Core kernel specific variables ***/
  long src8;
  long tv, fv, tv8, fv8, th8, mask, mask2;
  unsigned int th4, tv4, fv4;

  /* Replicating the values across a 64-bit word for SIMD operations */
  th4 = _pack2(thresh, thresh);
  th4 = _packl4(th4, th4);
  th8 = (uint64_t)_itoll(th4, th4);
  tv4 = _pack2(max_val, max_val);
  tv4 = _packl4(tv4, tv4);
  tv8 = (uint64_t)_itoll(tv4, tv4);
  fv4 = _pack2(min_val, min_val);
  fv4 = _packl4(fv4, fv4);
  fv8 = (uint64_t)_itoll(fv4, fv4);

  if (!evIN) { printf("Failed to alloc edmaIN handle.\n"); return; }

#ifdef TIDSP_OPENCL_VERBOSE
  printf ("\nEntering threshold!\n");
  clk_start = __clock();
#endif

  rows >>= 1;
  dest_ptr = (uchar *)dstptr;

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

#if 0
               T sdata = *(__global const T *)(srcptr + src_index);
                __global T * dst = (__global T *)(dstptr + dst_index);

                #ifdef THRESH_BINARY
                        dst[0] = sdata > (thresh) ? (T)(max_val) : (T)(0);
                #elif defined THRESH_BINARY_INV
                        dst[0] = sdata > (thresh) ? (T)(0) : (T)(max_val);
                #elif defined THRESH_TRUNC
                        dst[0] = clamp(sdata, (T)min_val, (T)(thresh));
                #elif defined THRESH_TOZERO
                        dst[0] = sdata > (thresh) ? sdata : (T)(0);
                #elif defined THRESH_TOZERO_INV
                        dst[0] = sdata > (thresh) ? (T)(0) : sdata;
                #endif

                gy++;
                src_index += src_step;
                dst_index += dst_step;
#endif
        /* Case 1B: All pointers are aligned to 8 byte boundaries, SIMD of 16 pixels at a time*/
        /* 3/16 = 0.1875 cycles per pixel */
        //#pragma UNROLL(2)
        for( i = 0; i < (cols / 8) ; i++ ) {
           src8 = _amem8_const(&yprev_ptr[i*8]);
           #ifdef THRESH_BINARY
                  mask = (long)_dxpnd4(_dcmpgtu4(src8, th8));
                  tv = mask & tv8;
                  _amem8(&dest_ptr[i*8]) = tv;
           #elif defined THRESH_BINARY_INV
                  mask = (long)_dxpnd4(_dcmpgtu4(th8, src8));
                  _amem8(&dest_ptr[i*8]) = mask & tv8;
           #elif defined THRESH_TRUNC
                  mask = (long)_dxpnd4(_dcmpgtu4(src8, th8));
                  tv = mask & tv8;
                  mask2 = (long)_dxpnd4(_dcmpgtu4(fv8, src8));
                  fv = mask2 & fv8;
                  tv = tv | fv;
                  tv = tv | ((~mask) | (~mask2)) & src8);
                  _amem8(&dest_ptr[i*8]) = tv;
           #elif defined THRESH_TOZERO
                  mask = (long)_dxpnd4(_dcmpgtu4(src8, th8));
                  _amem8(&dest_ptr[i*8]) = mask & src8;
           #elif defined THRESH_TOZERO_INV
                  mask = (long)_dxpnd4(_dcmpgtu4(src8, th8));
                  _amem8(&dest_ptr[i*8]) = (~mask) & src8;
           #endif
        }
    dest_ptr += cols;
  }
  EdmaMgr_wait(evIN);
  EdmaMgr_free(evIN);
#ifdef TIDSP_OPENCL_VERBOSE
  clk_end = __clock();
  printf ("TIDSP threshold clockdiff=%d\n", clk_end - clk_start);
#endif
}
/********************************************************************************************/
#else

////////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Zhang Ying, zhangying913@gmail.com
//
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
//M*/

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__kernel void threshold(__global const uchar * srcptr, int src_step, int src_offset,
                        __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols,
                        T1 thresh, T1 max_val, T1 min_val)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1) * STRIDE_SIZE;

    if (gx < cols)
    {
        int src_index = mad24(gy, src_step, mad24(gx, (int)sizeof(T), src_offset));
        int dst_index = mad24(gy, dst_step, mad24(gx, (int)sizeof(T), dst_offset));

        #pragma unroll
        for (int i = 0; i < STRIDE_SIZE; i++)
        {
            if (gy < rows)
            {
                T sdata = *(__global const T *)(srcptr + src_index);
                __global T * dst = (__global T *)(dstptr + dst_index);

                #ifdef THRESH_BINARY
                        dst[0] = sdata > (thresh) ? (T)(max_val) : (T)(0);
                #elif defined THRESH_BINARY_INV
                        dst[0] = sdata > (thresh) ? (T)(0) : (T)(max_val);
                #elif defined THRESH_TRUNC
                        dst[0] = clamp(sdata, (T)min_val, (T)(thresh));
                #elif defined THRESH_TOZERO
                        dst[0] = sdata > (thresh) ? sdata : (T)(0);
                #elif defined THRESH_TOZERO_INV
                        dst[0] = sdata > (thresh) ? (T)(0) : sdata;
                #endif

                gy++;
                src_index += src_step;
                dst_index += dst_step;
            }
        }
    }
}
#endif

