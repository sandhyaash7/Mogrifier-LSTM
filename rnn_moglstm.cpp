//#include "rnn_header.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include "hls_math.h"
#include <ap_fixed.h>
#include <ap_int.h>

#define n 7000
#define m 64
#define s 6
#define m2 128
#define m4 256

#define np 3
#define mp 64
#define sp 6

#define mogrify_steps 5 //defines how many mogrifying layers you want

typedef ap_fixed<14,2> ftype;
typedef ap_fixed<14,6> dtype;
typedef ap_fixed<9,9> intype;


dtype tanhf(dtype A){
	return hls::sinh(A)/hls::cosh(A);
}


void rnn_moglstm(ftype w_all[m2][m4], ftype b_all[m4], dtype h[m], dtype c[m],  dtype inputs[m2], ftype mogrifier_weights[mogrify_steps][m], ftype mogrifier_bias_even[mogrify_steps], ftype mogrifier_bias_odd[mogrify_steps][m])
{
    dtype gates[m4];
    dtype fi[m], fi2[m];
    dtype fC[m];
    dtype ff[m], ff2[m];
    dtype fo[m], fo2[m];
    dtype relu1;
    dtype relu2;

    dtype temp1;
    dtype temp2;
    dtype arr[m];

	#pragma HLS array_partition variable=gates complete dim=1
	#pragma HLS array_partition variable= ff cyclic factor=64 dim=1
	#pragma HLS array_partition variable= fi cyclic factor=64 dim=1
	#pragma HLS array_partition variable= fo cyclic factor=64 dim=1
	#pragma HLS array_partition variable= ff2 cyclic factor=64 dim=1
	#pragma HLS array_partition variable= fi2 cyclic factor=64 dim=1
	#pragma HLS array_partition variable= fo2 cyclic factor=64 dim=1
	#pragma HLS array_partition variable= fC cyclic factor=64 dim=1
	#pragma HLS array_partition variable= arr cyclic factor=64 dim=1


	//mogrifier inputs to be called---- weights 0 through 4 of mogrifier_list
	//biases 0 through 4 of mogrifier_list


	for (int i=0;i<mogrify_steps;i++){
		if((i+1)%2==0){
			for (int b=0;b<m;b+=1){
//#pragma HLS unroll factor=32
#pragma HLS pipeline II = 1
				relu1 = inputs[b]*mogrifier_weights[i][b]+mogrifier_bias_even[i];
				relu1 = 2* relu1 *h[b];
				//sigmoid1 =
				//h[b]=2*(dtype)(1/(1+(dtype)hls::exp(inputs[b]*mogrifier_weights[i][b]+mogrifier_bias_even[i])))*h[b];
				h[b] = max(0,relu1);

			}
		}
		else{
			for (int a=0;a<m;a+=1){
#pragma HLS unroll factor=32
#pragma HLS pipeline II=1
				relu2 = h[a]*mogrifier_weights[i][a]+mogrifier_bias_odd[i][a];
				relu2 = 2*relu2*inputs[a];
				//h[a]=2* sigmoid(h[a]*mogrifier_weights[i][a]+mogrifier_bias_odd[i][a]) *inputs[a];
				inputs[a]=max(0,relu2);
			}

		}
	}



//inputs and hidden state updation

    // Forward pass through LSTM layers
    big_loop1:for(int j=0; j<m4; j=j+1){
#pragma HLS unroll factor=16
        dtype gates_tmp = b_all[j];
        dtype gates_tmp2 = 0;
        dtype gates_tmp3 = 0;
        dtype gates_tmp4 = 0;
#pragma HLS pipeline II=1
        big_loop2:for(int k=0; k<m2/4; k=k+1){
            gates_tmp=gates_tmp+inputs[k]*w_all[k][j];
        }
        big_loop3:for(int k=m2/4; k<m2/2; k=k+1){
            gates_tmp2=gates_tmp2+inputs[k]*w_all[k][j];
        }
        big_loop4:for(int k=m2/2; k<((m2/2) + (m2/4)); k=k+1){
            gates_tmp3=gates_tmp3+inputs[k]*w_all[k][j];
        }
        big_loop5:for(int k=(m2/2) + (m2/4); k<m2; k=k+1){
            gates_tmp4=gates_tmp4+inputs[k]*w_all[k][j];
        }
        gates[j] = gates_tmp + gates_tmp2 + gates_tmp3 + gates_tmp4;
    }

    for(int j=0; j<m; j=j+1){
#pragma HLS unroll factor=32
#pragma HLS pipeline II=1
        ff[j]=(dtype)(1+hls::exp((dtype)-(gates[j+2*m]+1)));
        fi[j]=(dtype)(1+hls::exp((dtype)-(gates[j])));
        fo[j]=(dtype)(1+hls::exp((dtype)-(gates[j+3*m])));
    }

    for(int j=0; j<m; j=j+1){
#pragma HLS unroll factor=32
#pragma HLS pipeline II=1
        fC[j]=(dtype)tanhf((dtype)gates[j+m]);
    }

    for(int j=0; j<m; j=j+1){
#pragma HLS unroll factor=32
#pragma HLS pipeline II=1
        ff2[j]=(dtype)1.0/ff[j];
        fi2[j]=(dtype)1.0/fi[j];
        fo2[j]=(dtype)1.0/fo[j];
    }


    for(int j=0;j<m;j=j+1){
#pragma HLS unroll factor=32
#pragma HLS pipeline II=1
        	temp1=c[j]*ff2[j];
        	temp2=fi2[j]*fC[j];
        	arr[j]=temp1+temp2;
        }

        for(int j=0;j<m;j=j+1){
#pragma HLS unroll factor=32
#pragma HLS pipeline II=1
        	c[j]=(dtype)tanhf((dtype)(arr[j]));
        }

        for(int j=0;j<m;j=j+1){
#pragma HLS unroll factor=32
#pragma HLS pipeline II=1
        	h[j]=c[j]*fo2[j];
        }
}
