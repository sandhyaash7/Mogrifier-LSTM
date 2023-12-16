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

#define mogrify_steps 5

typedef ap_fixed<14,2> ftype;
typedef ap_fixed<14,6> dtype;
typedef ap_fixed<9,9> intype;

void rnn_model(intype signals[n][m], ftype weights_in[m][m], ftype biases_in[m], ftype w_hidd2[m][m], ftype b_hidd2[m],
		ftype w_hidd3[m][m], ftype b_hidd3[m], ftype w_all[m2][m4], ftype b_all[m4],
		ftype w_all2[m2][m4], ftype b_all2[m4], ftype w_out[m][s], ftype b_out[s], int cl[n],ftype mogrifier_weights[mogrify_steps][m], ftype mogrifier_bias_even[mogrify_steps], ftype mogrifier_bias_odd[mogrify_steps][m]);



int main(){

    printf("hi");
	//int v=0;

	//LAYER IN
    //weights
    ftype weights_in[m][m];
    std::ifstream file1("/home/user/Downloads/w_in.csv");

    for(int row = 0; row < m; ++row)
    {
        std::string line;
        std::getline(file1, line);
        std::stringstream iss(line);

        for (int col = 0; col < m; ++col)
        {
            std::string val;
            std::getline(iss, val, ',');
            //if ( !iss.good() )
              //  break;

            std::stringstream convertor(val);
            convertor >> weights_in[row][col];
        }
    }


    //biases
    ftype biases_in[m];
    std::ifstream file2("/home/user/Downloads/b_in.csv");

    for(int row = 0; row < m; ++row)
    {
        std::string line;
        std::getline(file2, line);
        std::stringstream iss(line);
        std::stringstream convertor(line);
        convertor >> biases_in[row];

    }


   //LAYER 2

    ftype w_hidd2[m][m];
    std::ifstream file3("/home/user/Downloads/w_hidd2.csv");

    for(int row = 0; row < m; ++row)
    {
        std::string line;
        std::getline(file3, line);
        std::stringstream iss(line);

        for (int col = 0; col < m; ++col)
        {
            std::string val;
            std::getline(iss, val, ',');
            //if ( !iss.good() )
              //  break;

            std::stringstream convertor(val);
            convertor >> w_hidd2[row][col];
        }
    }


    //biases
    ftype b_hidd2[m];
    std::ifstream file4("/home/user/Downloads/b_hidd2.csv");

    for(int row = 0; row < m; ++row)
    {
        std::string line;
        std::getline(file4, line);
        std::stringstream iss(line);
        std::stringstream convertor(line);
        convertor >> b_hidd2[row];

    }


    //LAYER 3

    ftype w_hidd3[m][m];
    std::ifstream file5("/home/user/Downloads/w_hidd3.csv");

    for(int row = 0; row < m; ++row)
    {
        std::string line;
        std::getline(file5, line);
        std::stringstream iss(line);

        for (int col = 0; col < m; ++col)
        {
            std::string val;
            std::getline(iss, val, ',');
            //if ( !iss.good() )
              //  break;

            std::stringstream convertor(val);
            convertor >> w_hidd3[row][col];
        }
    }


    //biases
    ftype b_hidd3[m];
    std::ifstream file6("/home/user/Downloads/b_hidd3.csv");

    for(int row = 0; row < m; ++row)
    {
        std::string line;
        std::getline(file6, line);
        std::stringstream iss(line);
        std::stringstream convertor(line);
        convertor >> b_hidd3[row];

    }


	//MOG LSTM LAYER 


	//weights of mog lstm 
    ftype w_all[m2][m4];
    std::ifstream file7("/home/user/Downloads/weights_all.csv");

    for(int row = 0; row < m2; ++row)
    {
        std::string line;
        std::getline(file7, line);
        std::stringstream iss(line);

        for (int col = 0; col < m4; ++col)
        {
            std::string val;
            std::getline(iss, val, ',');
            //if ( !iss.good() )
              //  break;

            std::stringstream convertor(val);
            convertor >> w_all[row][col];
        }
    }


	//biases of mog lstm
    ftype b_all[m4];
    std::ifstream file8("/home/user/Downloads/biases_all.csv");

    for(int row = 0; row < m4; ++row)
    {
        std::string line;
        std::getline(file8, line);
        std::stringstream iss(line);
        std::stringstream convertor(line);
        convertor >> b_all[row];

    }
  
//Mogrifier Parameter Weights
  ftype mogrifier_weights[mogrify_steps][m];
    std::ifstream file14("/home/user/Downloads/mog_weights.csv");

    for(int row = 0; row < mogrify_steps; ++row)
    {
        std::string line;
        std::getline(file14, line);
        std::stringstream iss(line);

        for (int col = 0; col < m; ++col)
        {
            std::string val;
            std::getline(iss, val, ',');
            //if ( !iss.good() )
              //  break;

            std::stringstream convertor(val);
            convertor >> mogrifier_weights[row][col];
        }
    }

  //Mogrifier Parameter Odd Biases
  ftype mogrifier_bias_odd[mogrify_steps][m];
    std::ifstream file15("/home/user/Downloads/mog_bias_odd.csv");

    for(int row = 0; row < mogrify_steps; ++row)
    {
        std::string line;
        std::getline(file15, line);
        std::stringstream iss(line);

        for (int col = 0; col < m; ++col)
        {
            std::string val;
            std::getline(iss, val, ',');
            //if ( !iss.good() )
              //  break;

            std::stringstream convertor(val);
            convertor >> mogrifier_bias_odd[row][col];
        }
    }

  //Mogrifier Parameter Even Biases
    ftype mogrifier_bias_even[m];
    std::ifstream file16("/home/user/Downloads/mog_bias_even.csv");

    for(int row = 0; row < m; ++row)
    {
        std::string line;
        std::getline(file16, line);
        std::stringstream iss(line);
        std::stringstream convertor(line);
        convertor >> mogrifier_bias_even[row];

    }

	

    //LAYER OUT

    //weights out
    ftype w_out[m][s];
    std::ifstream file11("/home/user/Downloads/w_out.csv");

    for(int row = 0; row < m; ++row)
    {
        std::string line;
        std::getline(file11, line);
        std::stringstream iss(line);

        for (int col = 0; col < s; ++col)
        {
            std::string val;
            std::getline(iss, val, ',');
            std::stringstream convertor(val);
            convertor >> w_out[row][col];
        }
    }


	//biases_out
    ftype b_out[s];
    std::ifstream file12("/home/user/Downloads/b_out.csv");

    for(int row = 0; row < s; ++row)
    {
        std::string line;
        std::getline(file12, line);
        std::stringstream iss(line);
        std::stringstream convertor(line);
        convertor >> b_out[row];

    }

    //Signals Input
    intype signals[n][m];
    std::ifstream file("/home/user/Downloads/data_testing.csv");
    for(int row = 0; row < n; ++row)
        {
        	std::string line;
            std::getline(file, line);
            std::stringstream iss(line);

            for (int col = 0; col < m; ++col)
            {

                std::string val;
                std::getline(iss, val, ',');
                //if ( !iss.good() )
                  //  break;

                std::stringstream convertor(val);
                convertor >> signals[row][col];
            }

    }
    //Real labels of input
    int labels[n];
    std::ifstream file13("/home/user/Downloads/labels_testing.csv");

    for(int row = 0; row < n; ++row)
    {
            std::string line;
            std::getline(file13, line);
            std::stringstream iss(line);
            std::stringstream convertor(line);
            convertor >> labels[row];

    }

	   //CALL FUNCTION
	   int cl[n];
	   int corr=0;


	   rnn_model(signals, weights_in, biases_in, w_hidd2, b_hidd2,
    		w_hidd3, b_hidd3, w_all, b_all, w_all2, b_all2, w_out, b_out, cl);

		for (int i=0;i<n;i=i+1){
			if(cl[i]==labels[i]){
				corr=corr+1;
			}
		}

    //}

	//ACCURACY
    float accuracy=(float)corr*100/n;
	printf("Accuracy\n");
	std::cout << accuracy <<std::endl;


}

