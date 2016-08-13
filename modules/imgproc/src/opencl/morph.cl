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

__attribute__((reqd_work_group_size(1,1,1))) __kernel void tidsp_morph_erode (__global const uchar * srcptr, int src_step, int src_offset,
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
  EdmaMgr_Handle evIN  = EdmaMgr_alloc(3);
  unsigned int srcPtr[3], dstPtr[3], numBytes[3];
  local uchar img_lines[4*MAX_LINE_SIZE];
  int clk_start, clk_end;
  int i, j;

  long dupper_word, dmid_word, dlower_word, dres0;
  unsigned int upper_word, mid_word, lower_word;
  unsigned int res0, res1;
  unsigned int dst_pixelU, dst_pixelL;
  unsigned int dst_pixelUu, dst_pixelLu;
  unsigned int byte0, byte1, byte2, byte3;
  unsigned int byte4, byte5;

  if (!evIN) { printf("Failed to alloc edmaIN handle.\n"); return; }

  clk_start = __clock();
  rows >>= 1;
  dest_ptr = (uchar *)dstptr;

  if(gid == 0)
  { /* Upper half of image */
    memset (img_lines, 0, 2 * MAX_LINE_SIZE);
    EdmaMgr_copy1D1D(evIN, (void*)(srcptr), (void*)(img_lines + 2 * MAX_LINE_SIZE), cols);
    fetch_rd_idx = cols;
  } else if(gid == 1)
  { /* Bottom half of image */
    srcPtr[0] = (unsigned int)(srcptr + (rows - 2) * cols);
    srcPtr[1] = (unsigned int)(srcptr + (rows - 1) * cols);
    srcPtr[2] = (unsigned int)(srcptr + (rows - 0) * cols);
    dstPtr[0] = (unsigned int)(img_lines + 0 * MAX_LINE_SIZE);  
    dstPtr[1] = (unsigned int)(img_lines + 1 * MAX_LINE_SIZE);  
    dstPtr[2] = (unsigned int)(img_lines + 2 * MAX_LINE_SIZE);  
    numBytes[0] = numBytes[1] = numBytes[2] = cols;
    EdmaMgr_copy1D1DLinked(evIN, srcPtr, dstPtr, numBytes, 3);
    fetch_rd_idx = (rows + 1) * cols;
    dest_ptr += rows * cols;
  } else return;
  start_rd_idx = 0;

  for (int y = 0; y < rows; y ++)
  {
    EdmaMgr_wait(evIN);
    rd_idx  = start_rd_idx;
    yprev_ptr = (uchar *)&img_lines[rd_idx];
    rd_idx = (rd_idx + MAX_LINE_SIZE) & (4*MAX_LINE_SIZE - 1);
    start_rd_idx = rd_idx;
    y_ptr     = (uchar *)&img_lines[rd_idx];
    rd_idx = (rd_idx + MAX_LINE_SIZE) & (4*MAX_LINE_SIZE - 1);
    ynext_ptr = (uchar *)&img_lines[rd_idx];
    rd_idx = (rd_idx + MAX_LINE_SIZE) & (4*MAX_LINE_SIZE - 1);
    EdmaMgr_copyFast(evIN, (void*)(srcptr + fetch_rd_idx), (void*)(&img_lines[rd_idx]));
    fetch_rd_idx += cols;
#pragma unroll 2
    for (i = 0; i < cols; i += 2) {
#if 1
        upper_word      = _mem4_const(&yprev_ptr[i]);
        mid_word        = _mem4_const(&y_ptr[i]);
        lower_word      = _mem4_const(&ynext_ptr[i]);
        res0            = _minu4(lower_word, _minu4 (upper_word, mid_word));
        byte3           = _extu(res0,  0, 24);
        byte2           = _extu(res0,  8, 24);
        byte1           = _extu(res0, 16, 24);
        byte0           = _extu(res0, 24, 24);
        dst_pixelL      = _min2(byte0, _min2(byte2, byte1));
        dst_pixelU      = _min2(byte1, _min2(byte3, byte2));
        _mem2(&dest_ptr[i]) = (unsigned short)((dst_pixelU << 8) | dst_pixelL);
#endif
#if 0
        dupper_word      = _mem8_const(&yprev_ptr[i]);
        dmid_word        = _mem8_const(&y_ptr[i]);
        dlower_word      = _mem8_const(&ynext_ptr[i]);
        dres0            = _dminu4(dupper_word, dmid_word);
        dres0            = _dminu4(dres0, dlower_word);
        res0             = _lo(dres0);
        byte3            = _extu(res0,  0, 24);
        byte2            = _extu(res0,  8, 24);
        byte1            = _extu(res0, 16, 24);
        byte0            = _extu(res0, 24, 24);
        dst_pixelL       = _min2(byte0, _min2(byte1, byte2));
        dst_pixelU       = _min2(byte1, _min2(byte2, byte3));
        res0             = _hi(dres0);
        byte5            = _extu(res0, 16, 24);
        byte4            = _extu(res0, 24, 24);
        dst_pixelLu      = _min2(byte2, _min2(byte3, byte4));
        dst_pixelUu      = _min2(byte3, _min2(byte4, byte5));
        _amem4(&dest_ptr[i]) = _packl4(_pack2(dst_pixelUu, dst_pixelLu), _pack2(dst_pixelU, dst_pixelLu));
#endif
    }
    dest_ptr += cols;
  }
  EdmaMgr_wait(evIN);
  EdmaMgr_free(evIN);

  clk_end = __clock();
  printf ("TIDSP erode clockdiff=%d\n", clk_end - clk_start);
}
/********************************************************************************************/
__attribute__((reqd_work_group_size(1,1,1))) __kernel void tidsp_morph_dilate (__global const uchar * srcptr, int src_step, int src_offset,
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
  EdmaMgr_Handle evIN  = EdmaMgr_alloc(3);
  unsigned int srcPtr[3], dstPtr[3], numBytes[3];
  local uchar img_lines[4*MAX_LINE_SIZE];
  int clk_start, clk_end;
  int i, j;
  unsigned int upper_word, mid_word, lower_word;
  unsigned int res0, res1, res2, res3, res4, res5;
  unsigned short dst_pixel;
  unsigned short dst_pixelL, dst_pixelU;
  unsigned int   byte0, byte1, byte2, byte3;

  /* -------------------------------------------------------------------- */
  /*  "Don't care" values in mask become '1's for the ORing step.  We     */
  /*  do this by converting negative values to "-1" (all 1s in binary)    */
  /*  and converting positive values to 0.                                */
  /* -------------------------------------------------------------------- */

  clk_start = __clock();

  if (!evIN) { printf("Failed to alloc edmaIN handle.\n"); return; }
  rows >>= 1;
  dest_ptr = (uchar *)dstptr;

  if(gid == 0)
  { /* Upper half of image */
    memset (img_lines, 0, 2 * MAX_LINE_SIZE);
    EdmaMgr_copy1D1D(evIN, (void*)(srcptr), (void*)(img_lines + 2 * MAX_LINE_SIZE), cols);
    fetch_rd_idx = cols;
  } else if(gid == 1)
  { /* Bottom half of image */
    srcPtr[0] = (unsigned int)(srcptr + (rows - 2) * cols);
    srcPtr[1] = (unsigned int)(srcptr + (rows - 1) * cols);
    srcPtr[2] = (unsigned int)(srcptr + (rows - 0) * cols);
    dstPtr[0] = (unsigned int)(img_lines + 0 * MAX_LINE_SIZE);  
    dstPtr[1] = (unsigned int)(img_lines + 1 * MAX_LINE_SIZE);  
    dstPtr[2] = (unsigned int)(img_lines + 2 * MAX_LINE_SIZE);  
    numBytes[0] = numBytes[1] = numBytes[2] = cols;
    EdmaMgr_copy1D1DLinked(evIN, srcPtr, dstPtr, numBytes, 3);
    fetch_rd_idx = (rows + 1) * cols;
    dest_ptr += rows * cols;
  } else return;
  start_rd_idx = 0;

  for (int y = 0; y < rows; y ++)
  {
    EdmaMgr_wait(evIN);
    rd_idx  = start_rd_idx;
    yprev_ptr = (uchar *)&img_lines[rd_idx];
    rd_idx = (rd_idx + MAX_LINE_SIZE) & (4*MAX_LINE_SIZE - 1);
    start_rd_idx = rd_idx;
    y_ptr     = (uchar *)&img_lines[rd_idx];
    rd_idx = (rd_idx + MAX_LINE_SIZE) & (4*MAX_LINE_SIZE - 1);
    ynext_ptr = (uchar *)&img_lines[rd_idx];
    rd_idx = (rd_idx + MAX_LINE_SIZE) & (4*MAX_LINE_SIZE - 1);
    EdmaMgr_copyFast(evIN, (void*)(srcptr + fetch_rd_idx), (void*)(&img_lines[rd_idx]));
    fetch_rd_idx += cols;
#pragma unroll 2
    for (i = 0; i < cols; i += 2) {
#if 0
        upper_word      = _mem4_const(&yprev_ptr[i]);
        mid_word        = _mem4_const(&y_ptr[i]);
        lower_word      = _mem4_const(&ynext_ptr[i]);
        res0            = _dotpu4 (upper_word, 0x00010101);
        res1            = _dotpu4 (mid_word,   0x00010101);
        res2            = _dotpu4 (lower_word, 0x00010101);
        res3            = _dotpu4 (upper_word, 0x01010100);
        res4            = _dotpu4 (mid_word,   0x01010100);
        res5            = _dotpu4 (lower_word, 0x01010100);
        dst_pixel       = 0;
        if((res0 + res1 + res2) > 0) dst_pixel = 0x00ff;
        if((res3 + res4 + res5) > 0) dst_pixel |= 0xff00;
#else
        upper_word      = _mem4_const(&yprev_ptr[i]);
        mid_word        = _mem4_const(&y_ptr[i]);
        lower_word      = _mem4_const(&ynext_ptr[i]);
        res0            = _maxu4 (lower_word, _maxu4 (upper_word, mid_word));
        byte3           = _extu(res0,  0, 24);
        byte2           = _extu(res0,  8, 24);
        byte1           = _extu(res0, 16, 24);
        byte0           = _extu(res0, 24, 24);
        dst_pixelL      = _max2(byte0, _max2(byte1, byte2));
        dst_pixelU      = _max2(byte1, _max2(byte2, byte3));
        _mem2(&dest_ptr[i]) = (unsigned short)((dst_pixelU << 8) | dst_pixelL);
#endif
    }
    dest_ptr += cols;
  }
  EdmaMgr_wait(evIN);
  EdmaMgr_free(evIN);
  clk_end = __clock();
  printf ("TIDSP dilate clockdiff=%d\n", clk_end - clk_start);
}

