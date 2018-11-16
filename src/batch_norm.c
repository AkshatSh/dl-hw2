#include "uwnet.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#define EPSILON 0.001

matrix mean(matrix x, int spatial)
{
    matrix m = make_matrix(1, x.cols/spatial);
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            m.data[j/spatial] += x.data[i*x.cols + j];
        }
    }
    for(i = 0; i < m.cols; ++i){
        m.data[i] = m.data[i] / x.rows / spatial;
    }
    return m;
}

matrix variance(matrix x, matrix m, int spatial)
{
    matrix v = make_matrix(1, x.cols/spatial);
    // TODO: 7.1 - calculate variance
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            v.data[j/spatial] += pow((x.data[i*x.cols + j] - m.data[j/spatial]), 2.0);
        }
    }
    for(i = 0; i < m.cols; ++i){
        v.data[i] = v.data[i] / x.rows / spatial;
    }
    return v;
}

matrix normalize(matrix x, matrix m, matrix v, int spatial)
{
    matrix norm = make_matrix(x.rows, x.cols);
    // TODO: 7.2 - normalize array, norm = (x - mean) / sqrt(variance + eps)
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            float variance = v.data[j/spatial];
            float mean = m.data[j / spatial];
            float val = x.data[i * x.cols + j];
            float norm_val = (val - mean) / sqrt(variance + EPSILON);
            norm.data[i * x.cols + j] = norm_val;
        }
    }
    return norm;
    
}

matrix batch_normalize_forward(layer l, matrix x)
{
    float s = .1;
    int spatial = x.cols / l.rolling_mean.cols;
    if (x.rows == 1){
        return normalize(x, l.rolling_mean, l.rolling_variance, spatial);
    }
    matrix m = mean(x, spatial);
    matrix v = variance(x, m, spatial);

    matrix x_norm = normalize(x, m, v, spatial);

    scal_matrix(1-s, l.rolling_mean);
    axpy_matrix(s, m, l.rolling_mean);

    scal_matrix(1-s, l.rolling_variance);
    axpy_matrix(s, v, l.rolling_variance);

    free_matrix(m);
    free_matrix(v);

    free_matrix(l.x[0]);
    l.x[0] = x;

    return x_norm;
}


matrix delta_mean(matrix d, matrix variance, int spatial)
{
    matrix dm = make_matrix(1, variance.cols);
    // TODO: 7.3 - calculate dL/dmean
    int i, j;
    for(i = 0; i < d.rows; ++i){
        for(j = 0; j < d.cols; ++j){
            float val = d.data[i*d.cols + j];
            float variance_term = variance.data[j/spatial];

            dm.data[j/spatial] += (val * -1.0/ sqrt(variance_term+ EPSILON) );
        }
    }

    return dm;
}

matrix delta_variance(matrix d, matrix x, matrix mean, matrix variance, int spatial)
{
    matrix dv = make_matrix(1, variance.cols);
    // TODO: 7.4 - calculate dL/dvariance
    int i,j;
    for(i = 0; i < d.rows; ++i){
        for(j = 0; j < d.cols; ++j){
            float dx_val = d.data[i*d.cols + j];
            float x_val = x.data[i*x.cols + j];
            float mean_val = mean.data[j / spatial];
            
            float variance_val = variance.data[j/spatial];

            dv.data[j/spatial] += dx_val * (x_val - mean_val)* (-1.0 / 2.0 ) * pow(variance_val + EPSILON, -3.0/2.0);
        }
    }
    return dv;
}

matrix delta_batch_norm(matrix d, matrix dm, matrix dv, matrix mean, matrix variance, matrix x, int spatial)
{
    int i, j;
    matrix dx = make_matrix(d.rows, d.cols);
    // TODO: 7.5 - calculate dL/dx
    for(i = 0; i < dx.rows; ++i){
        for(j = 0; j < dx.cols; ++j){
            float dx_val = d.data[i*d.cols + j];
            float x_val = x.data[i*x.cols + j];
            float mean_val = mean.data[j / spatial];
            float dm_val = dm.data[j / spatial];
            float dv_val = dv.data[j / spatial];
            float m = dx.cols / spatial;
            
            float variance_val = variance.data[j/spatial];

            dx.data[i * dx.cols + j] = (
                dx_val *
                (1.0 / (sqrt(variance_val + EPSILON))) +
                2.0 * (x_val - mean_val) / (m) * dv_val +
                dm_val / (m) 
            );
        }
    }
    return dx;
}

matrix batch_normalize_backward(layer l, matrix d)
{
    int spatial = d.cols / l.rolling_mean.cols;
    matrix x = l.x[0];

    matrix m = mean(x, spatial);
    matrix v = variance(x, m, spatial);

    matrix dm = delta_mean(d, v, spatial);
    matrix dv = delta_variance(d, x, m, v, spatial);
    matrix dx = delta_batch_norm(d, dm, dv, m, v, x, spatial);

    free_matrix(m);
    free_matrix(v);
    free_matrix(dm);
    free_matrix(dv);

    return dx;
}
