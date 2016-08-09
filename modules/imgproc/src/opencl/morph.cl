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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#define noconvert

#if cn != 3
#define loadpix(addr) *(__global const T *)(addr)
#define storepix(val, addr)  *(__global T *)(addr) = val
#define TSIZE (int)sizeof(T)
#else
#define loadpix(addr) vload3(0, (__global const T1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1 *)(addr))
#define TSIZE ((int)sizeof(T1)*3)
#endif

#ifdef DEPTH_0
#define MIN_VAL 0
#define MAX_VAL UCHAR_MAX
#elif defined DEPTH_1
#define MIN_VAL SCHAR_MIN
#define MAX_VAL SCHAR_MAX
#elif defined DEPTH_2
#define MIN_VAL 0
#define MAX_VAL USHRT_MAX
#elif defined DEPTH_3
#define MIN_VAL SHRT_MIN
#define MAX_VAL SHRT_MAX
#elif defined DEPTH_4
#define MIN_VAL INT_MIN
#define MAX_VAL INT_MAX
#elif defined DEPTH_5
#define MIN_VAL (-FLT_MAX)
#define MAX_VAL FLT_MAX
#elif defined DEPTH_6
#define MIN_VAL (-DBL_MAX)
#define MAX_VAL DBL_MAX
#endif

#ifdef OP_ERODE
#define VAL MAX_VAL
#elif defined OP_DILATE
#define VAL MIN_VAL
#else
#error "Unknown operation"
#endif

#ifdef OP_ERODE
#if defined INTEL_DEVICE && defined DEPTH_0
#define MORPH_OP(A, B) ((A) < (B) ? (A) : (B))
#else
#define MORPH_OP(A, B) _min2((A), (B))
#endif
#endif
#ifdef OP_DILATE
#define MORPH_OP(A, B) _max2((A), (B))
#endif

#define PROCESS(y, x) \
    temp = LDS_DAT[mad24(l_y + y, width, l_x + x)]; \
    res = MORPH_OP(res, temp);

// BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii
#define ELEM(i, l_edge, r_edge, elem1, elem2) (i) < (l_edge) | (i) >= (r_edge) ? (elem1) : (elem2)

#if defined OP_GRADIENT || defined OP_TOPHAT || defined OP_BLACKHAT
#define EXTRA_PARAMS , __global const uchar * matptr, int mat_step, int mat_offset
#else
#define EXTRA_PARAMS
#endif

__kernel void morph(__global const uchar * srcptr, int src_step, int src_offset,
                    __global uchar * dstptr, int dst_step, int dst_offset,
                    int src_offset_x, int src_offset_y, int cols, int rows,
                    int src_whole_cols, int src_whole_rows EXTRA_PARAMS)
{
    int gidx = get_global_id(0), gidy = get_global_id(1);
    int l_x = get_local_id(0), l_y = get_local_id(1);
    int x = get_group_id(0) * LSIZE0, y = get_group_id(1) * LSIZE1;
    int start_x = x + src_offset_x - RADIUSX;
    int width = mad24(RADIUSX, 2, LSIZE0 + 1);
    int start_y = y + src_offset_y - RADIUSY;
    int point1 = mad24(l_y, LSIZE0, l_x);
    int point2 = point1 + LSIZE0 * LSIZE1;
    int tl_x = point1 % width, tl_y = point1 / width;
    int tl_x2 = point2 % width, tl_y2 = point2 / width;
    int cur_x = start_x + tl_x, cur_y = start_y + tl_y;
    int cur_x2 = start_x + tl_x2, cur_y2 = start_y + tl_y2;
    int start_addr = mad24(cur_y, src_step, cur_x * TSIZE);
    int start_addr2 = mad24(cur_y2, src_step, cur_x2 * TSIZE);

    __local T LDS_DAT[2 * LSIZE1 * LSIZE0];

    // read pixels from src
    int end_addr = mad24(src_whole_rows - 1, src_step, src_whole_cols * TSIZE);
    start_addr = start_addr < end_addr && start_addr > 0 ? start_addr : 0;
    start_addr2 = start_addr2 < end_addr && start_addr2 > 0 ? start_addr2 : 0;

    T temp0 = loadpix(srcptr + start_addr);
    T temp1 = loadpix(srcptr + start_addr2);

    // judge if read out of boundary
    temp0 = ELEM(cur_x, 0, src_whole_cols, (T)(VAL), temp0);
    temp0 = ELEM(cur_y, 0, src_whole_rows, (T)(VAL), temp0);

    temp1 = ELEM(cur_x2, 0, src_whole_cols, (T)(VAL), temp1);
    temp1 = ELEM(cur_y2, 0, src_whole_rows, (T)(VAL), temp1);

    LDS_DAT[point1] = temp0;
    LDS_DAT[point2] = temp1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gidx < cols && gidy < rows)
    {
        T res = (T)(VAL), temp;
        PROCESS_ELEMS;

        int dst_index = mad24(gidy, dst_step, mad24(gidx, TSIZE, dst_offset));

#if defined OP_GRADIENT || defined OP_TOPHAT || defined OP_BLACKHAT
        int mat_index =  mad24(gidy, mat_step, mad24(gidx, TSIZE, mat_offset));
        T value = loadpix(matptr + mat_index);

#ifdef OP_GRADIENT
        storepix(convertToT(convertToWT(res) - convertToWT(value)), dstptr + dst_index);
#elif defined OP_TOPHAT
        storepix(convertToT(convertToWT(value) - convertToWT(res)), dstptr + dst_index);
#elif defined OP_BLACKHAT
        storepix(convertToT(convertToWT(res) - convertToWT(value)), dstptr + dst_index);
#endif
#else // erode or dilate
        storepix(res, dstptr + dst_index);
#endif
    }
}
// Texas Instruments Inc 2016
// DSP Friendly implementation of erode and dilate functions 
// This is a subset of complete feature set available in generic baseline (above) kernels
//
#include <dsp_c.h>
#include <edmamgr.h>

#define MAX_LINE_SIZE 2048

__attribute__((reqd_work_group_size(2,1,1))) __kernel void tidsp_morph_erode (__global const uchar * srcptr, int src_step, int src_offset,
                    __global uchar * dstptr, int dst_step, int dst_offset,
                    int src_offset_x, int src_offset_y, int cols, int rows,
                    int src_whole_cols, int src_whole_rows)
{
  /* Binary input expected, 0 or 255 */
  /* IMG_erode_bin((const unsigned char *)srcptr, (const unsigned char *)dstptr, (const char *)mask, cols); */
  uchar *y_start_ptr = (uchar *)srcptr;
  uchar * restrict dest_ptr;
  uchar * restrict yprev_ptr;
  uchar * restrict y_ptr;
  uchar * restrict ynext_ptr;
  uchar result;
  int   rd_idx, start_rd_idx, fetch_rd_idx;
  int   gid   = get_global_id(0);
  EdmaMgr_Handle evIN  = EdmaMgr_alloc(1);
  local uchar img_lines[4*MAX_LINE_SIZE];
  int clk_start, clk_end;
  clk_start = __clock();
  int tot_pixels = 0;
  if (!evIN) { printf("Failed to alloc edmaIN handle.\n"); return; }
/**
  printf("TIDSP_erode: gid=%d clock_ticks:%08x\n", get_global_id(0), __clock());
**/
  rows >>= 1;
  dest_ptr = (uchar *)dstptr;

  if(gid == 0)
  { /* Upper half of image */
    memset (img_lines, 0, 2 * MAX_LINE_SIZE);
    EdmaMgr_copy1D1D(evIN, (void*)(srcptr), (void*)(img_lines + 2 * MAX_LINE_SIZE), cols);
    fetch_rd_idx = cols;
  } else if(gid == 1)
  { /* Bottom half of image */
    EdmaMgr_copy1D1D(evIN, (void*)(srcptr + (rows - 2) * cols), (void*)(img_lines), cols);
    EdmaMgr_wait(evIN);
    EdmaMgr_copy1D1D(evIN, (void*)(srcptr + (rows - 1) * cols), (void*)(img_lines + MAX_LINE_SIZE), cols);
    EdmaMgr_wait(evIN);
    EdmaMgr_copy1D1D(evIN, (void*)(srcptr + rows * cols), (void*)(img_lines + 2 * MAX_LINE_SIZE), cols);
    fetch_rd_idx = (rows + 1) * cols;
    dest_ptr += rows * cols;
  } else return;
  start_rd_idx = 0;

  for (int y = 0; y < rows; y ++)
  {
    EdmaMgr_wait(evIN);
    rd_idx  = start_rd_idx;
    yprev_ptr = (uchar *)&img_lines[rd_idx];
    rd_idx = (rd_idx + MAX_LINE_SIZE) & (MAX_LINE_SIZE - 1);
    start_rd_idx = rd_idx;
    y_ptr     = (uchar *)&img_lines[rd_idx];
    rd_idx = (rd_idx + MAX_LINE_SIZE) & (MAX_LINE_SIZE - 1);
    ynext_ptr = (uchar *)&img_lines[rd_idx];
    rd_idx = (rd_idx + MAX_LINE_SIZE) & (MAX_LINE_SIZE - 1);
    EdmaMgr_copy1D1D(evIN, (void*)(srcptr + fetch_rd_idx), (void*)(&img_lines[rd_idx]), cols);
    fetch_rd_idx += cols;

    for (int x = 0; x < cols; x ++)
    { /* Input pixels are either 255 or 0, i.e. binary */ 
       result  = *yprev_ptr ++;
       result &= *yprev_ptr ++;
       result &= *yprev_ptr --;
       result &= *y_ptr ++;
       result &= *y_ptr ++;
       result &= *y_ptr --;
       result &= *ynext_ptr ++;
       result &= *ynext_ptr ++;
       result &= *ynext_ptr --;
       *dest_ptr ++ = result;
       tot_pixels ++;
    }
  }
  EdmaMgr_wait(evIN);
  EdmaMgr_free(evIN);

  clk_end = __clock();
  printf ("TIDSP clockdiff=%d pixels=%d\n", clk_end - clk_start, tot_pixels);
}
/********************************************************************************************/
__attribute__((reqd_work_group_size(2,1,1))) __kernel void tidsp_morph_dilate (__global const uchar * srcptr, int src_step, int src_offset,
                    __global uchar * dstptr, int dst_step, int dst_offset,
                    int src_offset_x, int src_offset_y, int cols, int rows,
                    int src_whole_cols, int src_whole_rows)
{
  /* Binary input expected, 0 or 255 */
  /* IMG_erode_bin((const unsigned char *)srcptr, (const unsigned char *)dstptr, (const char *)mask, cols); */
  uchar *y_start_ptr = (uchar *)srcptr;
  uchar * restrict dest_ptr;
  uchar * restrict yprev_ptr;
  uchar * restrict y_ptr;
  uchar * restrict ynext_ptr;
  uchar result;
  int   rd_idx, start_rd_idx, fetch_rd_idx;
  int   gid   = get_global_id(0);
  EdmaMgr_Handle evIN  = EdmaMgr_alloc(1);
  local uchar img_lines[4*MAX_LINE_SIZE];
  int tot_pixels = 0;
  int clk_start, clk_end;
  clk_start = __clock();

  if (!evIN) { printf("Failed to alloc edmaIN handle.\n"); return; }
/**
  printf("TIDSP_dilate: gid=%d clock_ticks:%08x\n", get_global_id(0), __clock());
**/
  rows >>= 1;
  dest_ptr = (uchar *)dstptr;

  if(gid == 0)
  { /* Upper half of image */
    memset (img_lines, 0, 2 * MAX_LINE_SIZE);
    EdmaMgr_copy1D1D(evIN, (void*)(srcptr), (void*)(img_lines + 2 * MAX_LINE_SIZE), cols);
    fetch_rd_idx = cols;
  } else if(gid == 1)
  { /* Bottom half of image */
    EdmaMgr_copy1D1D(evIN, (void*)(srcptr + (rows - 2) * cols), (void*)(img_lines), cols);
    EdmaMgr_wait(evIN);
    EdmaMgr_copy1D1D(evIN, (void*)(srcptr + (rows - 1) * cols), (void*)(img_lines + MAX_LINE_SIZE), cols);
    EdmaMgr_wait(evIN);
    EdmaMgr_copy1D1D(evIN, (void*)(srcptr + rows * cols), (void*)(img_lines + 2 * MAX_LINE_SIZE), cols);
    fetch_rd_idx = (rows + 1) * cols;
    dest_ptr += rows * cols;
  } else return;
  start_rd_idx = 0;

  for (int y = 0; y < rows; y ++)
  {
    EdmaMgr_wait(evIN);
    rd_idx  = start_rd_idx;
    yprev_ptr = (uchar *)&img_lines[rd_idx];
    rd_idx = (rd_idx + MAX_LINE_SIZE) & (MAX_LINE_SIZE - 1);
    start_rd_idx = rd_idx;
    y_ptr     = (uchar *)&img_lines[rd_idx];
    rd_idx = (rd_idx + MAX_LINE_SIZE) & (MAX_LINE_SIZE - 1);
    ynext_ptr = (uchar *)&img_lines[rd_idx];
    rd_idx = (rd_idx + MAX_LINE_SIZE) & (MAX_LINE_SIZE - 1);
    EdmaMgr_copy1D1D(evIN, (void*)(srcptr + fetch_rd_idx), (void*)(&img_lines[rd_idx]), cols);
    fetch_rd_idx += cols;

    for (int x = 0; x < cols; x ++)
    { /* Input pixels are either 255 or 0, i.e. binary */ 
       result  = *yprev_ptr ++;
       result |= *yprev_ptr ++;
       result |= *yprev_ptr --;
       result |= *y_ptr ++;
       result |= *y_ptr ++;
       result |= *y_ptr --;
       result |= *ynext_ptr ++;
       result |= *ynext_ptr ++;
       result |= *ynext_ptr --;
       *dest_ptr ++ = result;
       tot_pixels ++;
    }
  }
  EdmaMgr_wait(evIN);
  EdmaMgr_free(evIN);
  clk_end = __clock();
  printf ("TIDSP clockdiff=%d pixels=%d\n", clk_end - clk_start, tot_pixels);
}
