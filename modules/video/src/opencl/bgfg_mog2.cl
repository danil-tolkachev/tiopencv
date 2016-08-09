#ifdef TIDSP_MOG2

#if CN==1

#define T_MEAN float
#define F_ZERO (0.0f)
#define cnMode 1

#define frameToMean(a, b) (b) = *(a);
#define meanToFrame(a, b) *b = convert_uchar_sat(a);

inline float sum(float val)
{
    return val;
}

#else

#define T_MEAN float4
#define F_ZERO (0.0f, 0.0f, 0.0f, 0.0f)
#define cnMode 4

#define meanToFrame(a, b)\
    b[0] = convert_uchar_sat(a.x); \
    b[1] = convert_uchar_sat(a.y); \
    b[2] = convert_uchar_sat(a.z);

#define frameToMean(a, b)\
    b.x = a[0]; \
    b.y = a[1]; \
    b.z = a[2]; \
    b.w = 0.0f;

inline float sum(const float4 val)
{
    return (val.x + val.y + val.z);
}

#endif

#include <dsp_c.h>
#include <edmamgr.h>

#define SUBLINE_CACHE 64

__kernel void mog2_kernel(__global const uchar* frame, int frame_step, int frame_offset, int frame_row, int frame_col,  //uchar || uchar3
                          __global uchar* modesUsed,                                                                    //uchar
                          __global uchar* weight,                                                                       //float
                          __global uchar* mean,                                                                         //T_MEAN=float || float4
                          __global uchar* variance,                                                                     //float
                          __global uchar* fgmask, const int fgmask_step, const int fgmask_offset,                       //uchar
                          const float alphaT, const float alpha1, const float prune,
                          const float c_Tb, const float c_TB, float c_Tg, const float c_varMin,                         //constants
                          const float c_varMax, const float c_varInit, const float c_tau
#ifdef SHADOW_DETECT
                          , const uchar c_shadowVal
#endif
                          )
{
  int clk_start, clk_end, clk_tot;
  int outx, inx, x, y;
  int start_y, end_y;
  int dst_idx, src_idx, tmp_mode;
  float *_weight, *_variance, *_mean;
  int idx_step = frame_row * frame_col;
  int srcEDMA[(NMIXTURES+1)*3];
  int dstEDMA[3][(NMIXTURES+1)*3];
  int num_bytesEDMA[(NMIXTURES+1)*3];
  int ping_pong_in   = 0;
  int ping_pong_proc = 0;
  int ping_pong_out  = -1;
  local float line_weight[3 * (NMIXTURES+1) * SUBLINE_CACHE];
  local float line_variance[3 * (NMIXTURES+1) * SUBLINE_CACHE];
  local float line_mean[3 * (NMIXTURES+1) * SUBLINE_CACHE];

  EdmaMgr_Handle evIN   = EdmaMgr_alloc((NMIXTURES+1)*3);
  EdmaMgr_Handle evOUT  = EdmaMgr_alloc((NMIXTURES+1)*3);

    if(get_global_id(0) == 0) {
      start_y = 0;
      end_y   = frame_row >> 1;
    } else {
      start_y = frame_row >> 1;
      end_y   = frame_row;
    }
    for(tmp_mode = 0; tmp_mode <= NMIXTURES; tmp_mode ++)
    {
      dst_idx = tmp_mode * SUBLINE_CACHE;
      src_idx = tmp_mode * idx_step + y * frame_col + 0;
      srcEDMA[tmp_mode * 3 + 0] = (int)&weight[src_idx];
      srcEDMA[tmp_mode * 3 + 1] = (int)&variance[src_idx];
      srcEDMA[tmp_mode * 3 + 2] = (int)&mean[src_idx];
      dstEDMA[0][tmp_mode * 3 + 0] = (int)&line_weight[dst_idx];
      dstEDMA[0][tmp_mode * 3 + 1] = (int)&line_variance[dst_idx];
      dstEDMA[0][tmp_mode * 3 + 2] = (int)&line_mean[dst_idx];
      dstEDMA[1][tmp_mode * 3 + 0] = (int)&line_weight[dst_idx + (NMIXTURES+1) * SUBLINE_CACHE];
      dstEDMA[1][tmp_mode * 3 + 1] = (int)&line_variance[dst_idx + (NMIXTURES+1) * SUBLINE_CACHE];
      dstEDMA[1][tmp_mode * 3 + 2] = (int)&line_mean[dst_idx + (NMIXTURES+1) * SUBLINE_CACHE];
      dstEDMA[2][tmp_mode * 3 + 0] = (int)&line_weight[dst_idx + 2 * (NMIXTURES+1) * SUBLINE_CACHE];
      dstEDMA[2][tmp_mode * 3 + 1] = (int)&line_variance[dst_idx + 2 * (NMIXTURES+1) * SUBLINE_CACHE];
      dstEDMA[2][tmp_mode * 3 + 2] = (int)&line_mean[dst_idx + 2 * (NMIXTURES+1) * SUBLINE_CACHE];
      num_bytesEDMA[tmp_mode * 3 +0] = num_bytesEDMA[tmp_mode * 3 +1] = num_bytesEDMA[tmp_mode * 3 +2] = SUBLINE_CACHE * sizeof(float);
    }
    EdmaMgr_copy1D1DLinked(evIN, srcEDMA, dstEDMA[ping_pong_in], num_bytesEDMA, 3 * (NMIXTURES+1));
    ping_pong_in ++;
    if(ping_pong_in > 2) ping_pong_in = 0;

    for (y = start_y; y < end_y; y++)
    {
    for (outx = 0; outx < frame_col; outx+= SUBLINE_CACHE)
    {
    for(tmp_mode = 0; tmp_mode <= NMIXTURES; tmp_mode ++)
    {
      src_idx = tmp_mode * idx_step + y * frame_col + outx;
      srcEDMA[tmp_mode * 3 + 0] = (int)&weight[src_idx];
      srcEDMA[tmp_mode * 3 + 1] = (int)&variance[src_idx];
      srcEDMA[tmp_mode * 3 + 2] = (int)&mean[src_idx];
    }
    _weight   = (float *)&line_weight[ping_pong_proc * (NMIXTURES+1) * SUBLINE_CACHE];
    _variance = (float *)&line_variance[ping_pong_proc * (NMIXTURES+1) * SUBLINE_CACHE];
    _mean     = (float *)&line_mean[ping_pong_proc * (NMIXTURES+1) * SUBLINE_CACHE];
    ping_pong_proc ++;
    if(ping_pong_proc > 2) ping_pong_proc = 0;
    EdmaMgr_wait(evIN);
    EdmaMgr_copyLinkedFast(evIN, srcEDMA, dstEDMA[ping_pong_in], 3 * (NMIXTURES+1));
    ping_pong_in ++;
    if(ping_pong_in > 2) ping_pong_in = 0;

clk_start = __clock();
    for (inx = 0; inx < SUBLINE_CACHE; inx++)
    {
        x = outx + inx;
        __global const uchar* _frame = (frame + mad24(y, frame_step, mad24(x, CN, frame_offset)));
        T_MEAN pix;
        frameToMean(_frame, pix);
        
        uchar foreground = 255; // 0 - the pixel classified as background
        int loc_mode_idx;
        bool fitsPDF = false; //if it remains zero a new GMM mode will be added

        __global uchar* _modesUsed = modesUsed + inx;
        uchar nmodes = _modesUsed[0];

        float totalWeight = 0.0f;
        uchar mode = 0;
        loc_mode_idx = inx;
        for (; mode < nmodes; ++mode)
        {
            float c_weight = mad(alpha1, _weight[loc_mode_idx], prune);
            float c_var = _variance[loc_mode_idx];
            T_MEAN c_mean = _mean[loc_mode_idx];

            T_MEAN diff = c_mean - pix;
            float dist2 = dot(diff, diff);

            if (totalWeight < c_TB && dist2 < c_Tb * c_var)
                foreground = 0;

            if (dist2 < c_Tg * c_var)
            {
                fitsPDF = true;
                c_weight += alphaT;

                //float k = alphaT / c_weight;
                float k = alphaT * _rcpsp(c_weight);
                T_MEAN mean_new = mad((T_MEAN)-k, diff, c_mean);
                float variance_new  = clamp(mad(k, (dist2 - c_var), c_var), c_varMin, c_varMax);

                for (int i = mode; i > 0; --i)
                {
                    int prev_idx = loc_mode_idx - SUBLINE_CACHE;
                    if (c_weight < _weight[prev_idx])
                        break;

                    _weight[loc_mode_idx]   = _weight[prev_idx];
                    _variance[loc_mode_idx] = _variance[prev_idx];
                    _mean[loc_mode_idx]     = _mean[prev_idx];

                    loc_mode_idx = prev_idx;
                }

                _mean[loc_mode_idx]     = mean_new;
                _variance[loc_mode_idx] = variance_new;
                _weight[loc_mode_idx]   = c_weight; //update weight by the calculated value

                totalWeight += c_weight;

                mode ++;
                loc_mode_idx += SUBLINE_CACHE;
                break;
            }
            if (c_weight < -prune)
                c_weight = 0.0f;

            _weight[loc_mode_idx] = c_weight; //update weight by the calculated value
            totalWeight += c_weight;
            loc_mode_idx += SUBLINE_CACHE;
        }
        
        for (; mode < nmodes; ++mode)
        {
            float c_weight = mad(alpha1, _weight[loc_mode_idx], prune);

            if (c_weight < -prune)
            {
                c_weight = 0.0f;
                nmodes = mode;
                break;
            }
            _weight[loc_mode_idx] = c_weight; //update weight by the calculated value
            totalWeight += c_weight;
            loc_mode_idx += SUBLINE_CACHE;
        }

        if (0.f < totalWeight)
        {
            totalWeight = _rcpsp(totalWeight);
            loc_mode_idx = inx;
            for (int mode = 0; mode < nmodes; ++mode) {
                _weight[loc_mode_idx] *= totalWeight;
                loc_mode_idx += SUBLINE_CACHE;
            }
        }

        if (!fitsPDF)
        {
            uchar mode = nmodes == (NMIXTURES) ? (NMIXTURES) - 1 : nmodes++;
            int mode_idx = mad24(mode, SUBLINE_CACHE, inx);

            if (nmodes == 1)
                _weight[mode_idx] = 1.f;
            else
            {
                _weight[mode_idx] = alphaT;

                for (int i = inx; i < mode_idx; i += SUBLINE_CACHE)
                    _weight[i] *= alpha1;
            }

            for (int i = nmodes - 1; i > 0; --i)
            {
                int prev_idx = mode_idx - idx_step;
                float value_at_prev_idx = _weight[prev_idx];
                if (alphaT < value_at_prev_idx)
                    break;

                _weight[mode_idx]   = value_at_prev_idx;
                _variance[mode_idx] = _variance[prev_idx];
                _mean[mode_idx]     = _mean[prev_idx];

                mode_idx = prev_idx;
            }

            _mean[mode_idx] = pix;
            _variance[mode_idx] = c_varInit;
        }

        _modesUsed[0] = nmodes;
#ifdef SHADOW_DETECT
        if (foreground)
        {
            float tWeight = 0.0f;
            int loc_mode_idx = inx;

            for (uchar mode = 0; mode < nmodes; ++mode)
            {
                T_MEAN c_mean = _mean[loc_mode_idx];

                T_MEAN pix_mean = pix * c_mean;

                float numerator = sum(pix_mean);
                float denominator = dot(c_mean, c_mean);

                if (denominator == 0)
                    break;

                if (numerator <= denominator && numerator >= c_tau * denominator)
                {
                    float a = numerator * _rcpsp(denominator);

                    T_MEAN dD = mad(a, c_mean, -pix);

                    if (dot(dD, dD) < c_Tb * _variance[loc_mode_idx] * a * a)
                    {
                        foreground = c_shadowVal;
                        break;
                    }
                }

                tWeight += _weight[loc_mode_idx];
                if (tWeight > c_TB)
                    break;
                loc_mode_idx += SUBLINE_CACHE;
            }
        }
#endif
        __global uchar* _fgmask = fgmask + mad24(y, fgmask_step, x + fgmask_offset);
        *_fgmask = (uchar)foreground;
    } //inx
clk_end = __clock();
clk_tot += (unsigned int)(clk_end - clk_start);
    if(ping_pong_out < 0) {
      ping_pong_out = 0;
      EdmaMgr_copy1D1DLinked(evOUT, dstEDMA[ping_pong_out], srcEDMA, num_bytesEDMA, 3 * (NMIXTURES+1));
    } else {
      EdmaMgr_wait(evOUT);
      EdmaMgr_copyLinkedFast(evOUT, dstEDMA[ping_pong_out], srcEDMA, 3 * (NMIXTURES+1));
    }
    ping_pong_out ++;
    if(ping_pong_out > 2) ping_pong_out = 0;
    } //outx
    }

    EdmaMgr_wait(evOUT);
    EdmaMgr_free(evIN);
    EdmaMgr_free(evOUT);
    printf ("TIDSP Modified MOG2 clk=%d frame_row=%d frame_col=%d (%p %p %p)\n", clk_tot, frame_row, frame_col, line_weight, line_variance, line_mean);
}

__kernel void getBackgroundImage2_kernel(__global const uchar* modesUsed,
                                         __global const uchar* weight,
                                         __global const uchar* mean,
                                         __global uchar* dst, int dst_step, int dst_offset, int dst_row, int dst_col,
                                         float c_TB)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < dst_col && y < dst_row)
    {
        int pt_idx =  mad24(y, dst_col, x);

        __global const uchar* _modesUsed = modesUsed + pt_idx;
        uchar nmodes = _modesUsed[0];

        T_MEAN meanVal = (T_MEAN)F_ZERO;

        float totalWeight = 0.0f;
        __global const float* _weight = (__global const float*)weight;
        __global const T_MEAN* _mean = (__global const T_MEAN*)(mean);
        int idx_step = dst_row * dst_col;
        for (uchar mode = 0; mode < nmodes; ++mode)
        {
            int mode_idx = mad24(mode, idx_step, pt_idx);
            float c_weight = _weight[mode_idx];
            T_MEAN c_mean = _mean[mode_idx];

            meanVal = mad(c_weight, c_mean, meanVal);

            totalWeight += c_weight;

            if (totalWeight > c_TB)
                break;
        }

        if (0.f < totalWeight)
            meanVal *= _rcpsp(totalWeight); //meanVal = meanVal / totalWeight;
        else
            meanVal = (T_MEAN)(0.f);
        __global uchar* _dst = dst + mad24(y, dst_step, mad24(x, CN, dst_offset));
        meanToFrame(meanVal, _dst);
    }
}

#else

#if CN==1

#define T_MEAN float
#define F_ZERO (0.0f)
#define cnMode 1

#define frameToMean(a, b) (b) = *(a);
#define meanToFrame(a, b) *b = convert_uchar_sat(a);

inline float sum(float val)
{
    return val;
}

#else

#define T_MEAN float4
#define F_ZERO (0.0f, 0.0f, 0.0f, 0.0f)
#define cnMode 4

#define meanToFrame(a, b)\
    b[0] = convert_uchar_sat(a.x); \
    b[1] = convert_uchar_sat(a.y); \
    b[2] = convert_uchar_sat(a.z);

#define frameToMean(a, b)\
    b.x = a[0]; \
    b.y = a[1]; \
    b.z = a[2]; \
    b.w = 0.0f;

inline float sum(const float4 val)
{
    return (val.x + val.y + val.z);
}

#endif

__kernel void mog2_kernel(__global const uchar* frame, int frame_step, int frame_offset, int frame_row, int frame_col,  //uchar || uchar3
                          __global uchar* modesUsed,                                                                    //uchar
                          __global uchar* weight,                                                                       //float
                          __global uchar* mean,                                                                         //T_MEAN=float || float4
                          __global uchar* variance,                                                                     //float
                          __global uchar* fgmask, int fgmask_step, int fgmask_offset,                                   //uchar
                          float alphaT, float alpha1, float prune,
                          float c_Tb, float c_TB, float c_Tg, float c_varMin,                                           //constants
                          float c_varMax, float c_varInit, float c_tau
#ifdef SHADOW_DETECT
                          , uchar c_shadowVal
#endif
                          )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if( x < frame_col && y < frame_row)
    {
        __global const uchar* _frame = (frame + mad24(y, frame_step, mad24(x, CN, frame_offset)));
        T_MEAN pix;
        frameToMean(_frame, pix);

        uchar foreground = 255; // 0 - the pixel classified as background

        bool fitsPDF = false; //if it remains zero a new GMM mode will be added

        int pt_idx =  mad24(y, frame_col, x);
        int idx_step = frame_row * frame_col;

        __global uchar* _modesUsed = modesUsed + pt_idx;
        uchar nmodes = _modesUsed[0];

        float totalWeight = 0.0f;

        __global float* _weight = (__global float*)(weight);
        __global float* _variance = (__global float*)(variance);
        __global T_MEAN* _mean = (__global T_MEAN*)(mean);

        uchar mode = 0;
        for (; mode < nmodes; ++mode)
        {
            int mode_idx = mad24(mode, idx_step, pt_idx);
            float c_weight = mad(alpha1, _weight[mode_idx], prune);

            float c_var = _variance[mode_idx];

            T_MEAN c_mean = _mean[mode_idx];

            T_MEAN diff = c_mean - pix;
            float dist2 = dot(diff, diff);

            if (totalWeight < c_TB && dist2 < c_Tb * c_var)
                foreground = 0;

            if (dist2 < c_Tg * c_var)
            {
                fitsPDF = true;
                c_weight += alphaT;

                float k = alphaT / c_weight;
                T_MEAN mean_new = mad((T_MEAN)-k, diff, c_mean);
                float variance_new  = clamp(mad(k, (dist2 - c_var), c_var), c_varMin, c_varMax);

                for (int i = mode; i > 0; --i)
                {
                    int prev_idx = mode_idx - idx_step;
                    if (c_weight < _weight[prev_idx])
                        break;

                    _weight[mode_idx]   = _weight[prev_idx];
                    _variance[mode_idx] = _variance[prev_idx];
                    _mean[mode_idx]     = _mean[prev_idx];

                    mode_idx = prev_idx;
                }

                _mean[mode_idx]     = mean_new;
                _variance[mode_idx] = variance_new;
                _weight[mode_idx]   = c_weight; //update weight by the calculated value

                totalWeight += c_weight;

                mode ++;

                break;
            }
            if (c_weight < -prune)
                c_weight = 0.0f;

            _weight[mode_idx] = c_weight; //update weight by the calculated value
            totalWeight += c_weight;
        }

        for (; mode < nmodes; ++mode)
        {
            int mode_idx = mad24(mode, idx_step, pt_idx);
            float c_weight = mad(alpha1, _weight[mode_idx], prune);

            if (c_weight < -prune)
            {
                c_weight = 0.0f;
                nmodes = mode;
                break;
            }
            _weight[mode_idx] = c_weight; //update weight by the calculated value
            totalWeight += c_weight;
        }

        if (0.f < totalWeight)
        {
            totalWeight = 1.f / totalWeight;
            for (int mode = 0; mode < nmodes; ++mode)
                _weight[mad24(mode, idx_step, pt_idx)] *= totalWeight;
        }

        if (!fitsPDF)
        {
            uchar mode = nmodes == (NMIXTURES) ? (NMIXTURES) - 1 : nmodes++;
            int mode_idx = mad24(mode, idx_step, pt_idx);

            if (nmodes == 1)
                _weight[mode_idx] = 1.f;
            else
            {
                _weight[mode_idx] = alphaT;

                for (int i = pt_idx; i < mode_idx; i += idx_step)
                    _weight[i] *= alpha1;
            }

            for (int i = nmodes - 1; i > 0; --i)
            {
                int prev_idx = mode_idx - idx_step;
                if (alphaT < _weight[prev_idx])
                    break;

                _weight[mode_idx]   = _weight[prev_idx];
                _variance[mode_idx] = _variance[prev_idx];
                _mean[mode_idx]     = _mean[prev_idx];

                mode_idx = prev_idx;
            }

            _mean[mode_idx] = pix;
            _variance[mode_idx] = c_varInit;
        }

        _modesUsed[0] = nmodes;
#ifdef SHADOW_DETECT
        if (foreground)
        {
            float tWeight = 0.0f;

            for (uchar mode = 0; mode < nmodes; ++mode)
            {
                int mode_idx = mad24(mode, idx_step, pt_idx);
                T_MEAN c_mean = _mean[mode_idx];

                T_MEAN pix_mean = pix * c_mean;

                float numerator = sum(pix_mean);
                float denominator = dot(c_mean, c_mean);

                if (denominator == 0)
                    break;

                if (numerator <= denominator && numerator >= c_tau * denominator)
                {
                    float a = numerator / denominator;

                    T_MEAN dD = mad(a, c_mean, -pix);

                    if (dot(dD, dD) < c_Tb * _variance[mode_idx] * a * a)
                    {
                        foreground = c_shadowVal;
                        break;
                    }
                }

                tWeight += _weight[mode_idx];
                if (tWeight > c_TB)
                    break;
            }
        }
#endif
        __global uchar* _fgmask = fgmask + mad24(y, fgmask_step, x + fgmask_offset);
        *_fgmask = (uchar)foreground;
    }
}

__kernel void getBackgroundImage2_kernel(__global const uchar* modesUsed,
                                         __global const uchar* weight,
                                         __global const uchar* mean,
                                         __global uchar* dst, int dst_step, int dst_offset, int dst_row, int dst_col,
                                         float c_TB)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < dst_col && y < dst_row)
    {
        int pt_idx =  mad24(y, dst_col, x);

        __global const uchar* _modesUsed = modesUsed + pt_idx;
        uchar nmodes = _modesUsed[0];

        T_MEAN meanVal = (T_MEAN)F_ZERO;

        float totalWeight = 0.0f;
        __global const float* _weight = (__global const float*)weight;
        __global const T_MEAN* _mean = (__global const T_MEAN*)(mean);
        int idx_step = dst_row * dst_col;
        for (uchar mode = 0; mode < nmodes; ++mode)
        {
            int mode_idx = mad24(mode, idx_step, pt_idx);
            float c_weight = _weight[mode_idx];
            T_MEAN c_mean = _mean[mode_idx];

            meanVal = mad(c_weight, c_mean, meanVal);

            totalWeight += c_weight;

            if (totalWeight > c_TB)
                break;
        }

        if (0.f < totalWeight)
            meanVal = meanVal / totalWeight;
        else
            meanVal = (T_MEAN)(0.f);
        __global uchar* _dst = dst + mad24(y, dst_step, mad24(x, CN, dst_offset));
        meanToFrame(meanVal, _dst);
    }
}
#endif
