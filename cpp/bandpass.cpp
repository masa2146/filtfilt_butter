#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>

#include <complex>

using namespace std;

#define N 10 //The number of images which construct a time series for each pixel
#define PI 3.1415926535897932384626433832795

double *ComputeLP(int FilterOrder)
{
    double *NumCoeffs;
    int m;
    int i;

    NumCoeffs = (double *)calloc(FilterOrder + 1, sizeof(double));
    if (NumCoeffs == NULL)
        return (NULL);

    NumCoeffs[0] = 1;
    NumCoeffs[1] = FilterOrder;
    m = FilterOrder / 2;
    for (i = 2; i <= m; ++i)
    {
        NumCoeffs[i] = (double)(FilterOrder - i + 1) * NumCoeffs[i - 1] / i;
        NumCoeffs[FilterOrder - i] = NumCoeffs[i];
    }
    NumCoeffs[FilterOrder - 1] = FilterOrder;
    NumCoeffs[FilterOrder] = 1;

    return NumCoeffs;
}

double *ComputeHP(int FilterOrder)
{
    double *NumCoeffs;
    int i;

    NumCoeffs = ComputeLP(FilterOrder);
    if (NumCoeffs == NULL)
        return (NULL);

    for (i = 0; i <= FilterOrder; ++i)
        if (i % 2)
            NumCoeffs[i] = -NumCoeffs[i];

    return NumCoeffs;
}

double *TrinomialMultiply(int FilterOrder, double *b, double *c)
{
    int i, j;
    double *RetVal;

    RetVal = (double *)calloc(4 * FilterOrder, sizeof(double));
    if (RetVal == NULL)
        return (NULL);

    RetVal[2] = c[0];
    RetVal[3] = c[1];
    RetVal[0] = b[0];
    RetVal[1] = b[1];

    for (i = 1; i < FilterOrder; ++i)
    {
        RetVal[2 * (2 * i + 1)] += c[2 * i] * RetVal[2 * (2 * i - 1)] - c[2 * i + 1] * RetVal[2 * (2 * i - 1) + 1];
        RetVal[2 * (2 * i + 1) + 1] += c[2 * i] * RetVal[2 * (2 * i - 1) + 1] + c[2 * i + 1] * RetVal[2 * (2 * i - 1)];

        for (j = 2 * i; j > 1; --j)
        {
            RetVal[2 * j] += b[2 * i] * RetVal[2 * (j - 1)] - b[2 * i + 1] * RetVal[2 * (j - 1) + 1] +
                             c[2 * i] * RetVal[2 * (j - 2)] - c[2 * i + 1] * RetVal[2 * (j - 2) + 1];
            RetVal[2 * j + 1] += b[2 * i] * RetVal[2 * (j - 1) + 1] + b[2 * i + 1] * RetVal[2 * (j - 1)] +
                                 c[2 * i] * RetVal[2 * (j - 2) + 1] + c[2 * i + 1] * RetVal[2 * (j - 2)];
        }

        RetVal[2] += b[2 * i] * RetVal[0] - b[2 * i + 1] * RetVal[1] + c[2 * i];
        RetVal[3] += b[2 * i] * RetVal[1] + b[2 * i + 1] * RetVal[0] + c[2 * i + 1];
        RetVal[0] += b[2 * i];
        RetVal[1] += b[2 * i + 1];
    }
    return RetVal;
}

double *ComputeNumCoeffs(int FilterOrder, double Lcutoff, double Ucutoff, double *DenC)
{
    double *TCoeffs;
    double *NumCoeffs;
    std::complex<double> *NormalizedKernel;
    double Numbers[11] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int i;

    NumCoeffs = (double *)calloc(2 * FilterOrder + 1, sizeof(double));
    if (NumCoeffs == NULL)
        return (NULL);

    NormalizedKernel = (std::complex<double> *)calloc(2 * FilterOrder + 1, sizeof(std::complex<double>));
    if (NormalizedKernel == NULL)
        return (NULL);

    TCoeffs = ComputeHP(FilterOrder);
    if (TCoeffs == NULL)
        return (NULL);

    for (i = 0; i < FilterOrder; ++i)
    {
        NumCoeffs[2 * i] = TCoeffs[i];
        NumCoeffs[2 * i + 1] = 0.0;
    }
    NumCoeffs[2 * FilterOrder] = TCoeffs[FilterOrder];
    double cp[2];
    //double Bw;
    double Wn;
    cp[0] = 2 * 2.0 * tan(PI * Lcutoff / 2.0);
    cp[1] = 2 * 2.0 * tan(PI * Ucutoff / 2.0);

    //Bw = cp[1] - cp[0];
    //center frequency
    Wn = sqrt(cp[0] * cp[1]);
    Wn = 2 * atan2(Wn, 4);
    //double kern;
    const std::complex<double> result = std::complex<double>(-1, 0);

    for (int k = 0; k < 2 * FilterOrder + 1; k++)
    {
        NormalizedKernel[k] = std::exp(-sqrt(result) * Wn * Numbers[k]);
    }
    double b = 0;
    double den = 0;
    for (int d = 0; d < 2 * FilterOrder + 1; d++)
    {
        b += real(NormalizedKernel[d] * NumCoeffs[d]);
        den += real(NormalizedKernel[d] * DenC[d]);
    }
    for (int c = 0; c < 2 * FilterOrder + 1; c++)
    {
        NumCoeffs[c] = (NumCoeffs[c] * den) / b;
    }

    free(TCoeffs);
    return NumCoeffs;
}

double *ComputeDenCoeffs(int FilterOrder, double Lcutoff, double Ucutoff)
{
    int k;               // loop variables
    double theta;        // PI * (Ucutoff - Lcutoff)/2.0
    double cp;           // cosine of phi
    double st;           // sine of theta
    double ct;           // cosine of theta
    double s2t;          // sine of 2*theta
    double c2t;          // cosine 0f 2*theta
    double *RCoeffs;     // z^-2 coefficients
    double *TCoeffs;     // z^-1 coefficients
    double *DenomCoeffs; // dk coefficients
    double PoleAngle;    // pole angle
    double SinPoleAngle; // sine of pole angle
    double CosPoleAngle; // cosine of pole angle
    double a;            // workspace variables

    cp = cos(PI * (Ucutoff + Lcutoff) / 2.0);
    theta = PI * (Ucutoff - Lcutoff) / 2.0;
    st = sin(theta);
    ct = cos(theta);
    s2t = 2.0 * st * ct;       // sine of 2*theta
    c2t = 2.0 * ct * ct - 1.0; // cosine of 2*theta

    RCoeffs = (double *)calloc(2 * FilterOrder, sizeof(double));
    TCoeffs = (double *)calloc(2 * FilterOrder, sizeof(double));

    for (k = 0; k < FilterOrder; ++k)
    {
        PoleAngle = PI * (double)(2 * k + 1) / (double)(2 * FilterOrder);
        SinPoleAngle = sin(PoleAngle);
        CosPoleAngle = cos(PoleAngle);
        a = 1.0 + s2t * SinPoleAngle;
        RCoeffs[2 * k] = c2t / a;
        RCoeffs[2 * k + 1] = s2t * CosPoleAngle / a;
        TCoeffs[2 * k] = -2.0 * cp * (ct + st * SinPoleAngle) / a;
        TCoeffs[2 * k + 1] = -2.0 * cp * st * CosPoleAngle / a;
    }

    DenomCoeffs = TrinomialMultiply(FilterOrder, TCoeffs, RCoeffs);
    free(TCoeffs);
    free(RCoeffs);

    DenomCoeffs[1] = DenomCoeffs[0];
    DenomCoeffs[0] = 1.0;
    for (k = 3; k <= 2 * FilterOrder; ++k)
        DenomCoeffs[k] = DenomCoeffs[2 * k - 2];

    return DenomCoeffs;
}

void filter(int ord, double *a, double *b, int np, double *x, double *y)
{
    int i, j;
    y[0] = b[0] * x[0];
    for (i = 1; i < ord + 1; i++)
    {
        y[i] = 0.0;
        for (j = 0; j < i + 1; j++)
            y[i] = y[i] + b[j] * x[i - j];
        for (j = 0; j < i; j++)
            y[i] = y[i] - a[j + 1] * y[i - j - 1];
    }
    for (i = ord + 1; i < np + 1; i++)
    {
        y[i] = 0.0;
        for (j = 0; j < ord + 1; j++)
            y[i] = y[i] + b[j] * x[i - j];
        for (j = 0; j < ord; j++)
            y[i] = y[i] - a[j + 1] * y[i - j - 1];
    }
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    //Frequency bands is a vector of values - Lower Frequency Band and Higher Frequency Band

    //First value is lower cutoff and second value is higher cutoff
    //f1 = 0.5Gz f2=10Gz
    //fs=127Gz
    //Kotelnikov/2=Nyquist (127/2)
    double dt = 0.005000000000000;
    double flp = 0.1;
    double fhp = 1.5;
    int n = 4;
    double fs = 1 / (2 * dt);
    double FrequencyBands[2] = {flp/fs, fhp/fs}; //these values are as a ratio of f/fs, where fs is sampling rate, and f is cutoff frequency
    //and therefore should lie in the range [0 1]
    //Filter Order
    int FiltOrd = n;

    //Pixel Time Series
    /*int PixelTimeSeries[N];
    int outputSeries[N];
    */
    //Create the variables for the numerator and denominator coefficients
    double *DenC = 0;
    double *NumC = 0;
    //Pass Numerator Coefficients and Denominator Coefficients arrays into function, will return the same

    printf("\n");

    //is A in matlab function and the numbers are correct
    DenC = ComputeDenCoeffs(FiltOrd, FrequencyBands[0], FrequencyBands[1]);
    for (int k = 0; k < 2 * FiltOrd + 1; k++)
    {
        printf("DenC is: %lf\n", DenC[k]);
    }

    printf("\n");

    NumC = ComputeNumCoeffs(FiltOrd, FrequencyBands[0], FrequencyBands[1], DenC);
    for (int k = 0; k < 2 * FiltOrd + 1; k++)
    {
        printf("NumC is: %lf\n", NumC[k]);
    }

    double y[5];
    double x[5] = {1, 2, 3, 4, 5};
    filter(n, DenC, NumC, n, x, y);
        for (int k = 0; k < sizeof(x); k++)
    {
        printf("X is: %lf\n", y[k]);
    }
    return 1;
}