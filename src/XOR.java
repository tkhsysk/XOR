public class XOR {
	private double ALPHA = 1.2f;
	private double BETA = 1.2f;
	private int INPUT = 2;
	private int HIDDEN = 2;
	private int OUTPUT = 1;

	int PATTERN = 4; // パターンの種類
	int OUTER_CYCLES = 200; // 外部サイクル（一連のパターンの繰返し学習）の回数
	int INNER_CYCLES = 200; // 内部サイクル（同一パターンの繰返し学習）の回数

	private int[] sample_in = new int[INPUT+1];
	private int[] given_in = new int[INPUT+1];
	private int[] teach = new int[PATTERN];

	private double[][] weight_ih = new double[INPUT+1][HIDDEN];
	private double[] hidden_out = new double[HIDDEN+1];

	private double[][] weight_ho = new double[HIDDEN+1][OUTPUT];
	private double[] recog_out = new double[OUTPUT];

	int[][] sample_array = {{ 0, 0 , 1}, { 0, 1, 1}, { 1, 0, 1}, { 1, 1, 1}};
	int[][] teach_array = {{0}, {1}, {1}, {0}};
	
	
    public static void main(String[] args) {
    	XOR xor = new XOR();
    	xor.optPara();
    	
    	xor.given_in[0] = 1;
    	xor.given_in[1] = 0;
    	xor.given_in[2] = 1;
    	
    	xor.forwardNeuralNet(xor.given_in, xor.recog_out);
    	
    	System.out.println(xor.recog_out[0]);
    }
	
	void optPara(){
        //閾値と重みの乱数設定
        for(int j=0;j<HIDDEN;j++){
           for(int i=0;i<INPUT+1;i++)
              weight_ih[i][j]=(float)Math.random()-0.5f;
        }
        for(int k=0;k<OUTPUT;k++){
           for(int j=0;j<HIDDEN+1;j++)
              weight_ho[j][k]=(float)Math.random()-0.5f;
        }
		
		//学習
		for(int p=0;p<OUTER_CYCLES;p++){
			//patternの切り替え 0 or 1
			for(int q=0;q<PATTERN;q++){
				sample_in = sample_array[q];
				teach = teach_array[q];
				for(int r=0;r<INNER_CYCLES;r++){
					forwardNeuralNet(sample_in, recog_out);
					backwardNeuralNet();
				}
			}
		}
	}

	void forwardNeuralNet(int[] input, double[] output){
		double[] out = new double[OUTPUT];
		double[] hidden = new double[HIDDEN+1];
		
		//中間層
		for(int j=0; j<HIDDEN; j++){
			for(int i=0; i<INPUT+1; i++){
				hidden[j] += input[i]*weight_ih[i][j];
			}
			hidden_out[j] = sigmoid(hidden[j]);
		}
		hidden_out[HIDDEN] = 1.0;
		
		//出力層
		for(int k=0; k<OUTPUT; k++){
			for(int j=0; j<HIDDEN+1; j++){
				out[k] += hidden_out[j]*weight_ho[j][k];
			}
			output[k] = sigmoid(out[k]);
		}		
		
	}
	
	void backwardNeuralNet(){
		int i, j, k;
		
		double[] output_error = new double[OUTPUT];
		double[] hidden_error = new double[HIDDEN];
		double temp_error;
		
		//出力層
		for(k=0; k<OUTPUT; k++){
			output_error[k] = (teach[k] - recog_out[k]) * recog_out[k] * (1.0 - recog_out[k]);
		}
		//中間層
		for(j=0; j<HIDDEN; j++){
			temp_error = 0.0;
			for(k=0; k<OUTPUT; k++){
				temp_error += output_error[k] * weight_ho[j][k];
			}
			hidden_error[j] = hidden_out[j] * (1.0 - hidden_out[j]) * temp_error;
		}
		
		//重み補正
		for(k=0; k<OUTPUT; k++){
			for(j=0; j<HIDDEN+1; j++){
				weight_ho[j][k] += ALPHA * output_error[k] * hidden_out[j];
			}
		}
		for(j=0; j<HIDDEN; j++){
			for(i=0; i<INPUT+1; i++){
				weight_ih[i][j] += ALPHA * hidden_error[k] * sample_in[i];
			}
		}
	}
	
	double sigmoid(double x){
		return 1.0/(1.0+Math.exp(-BETA*x));
	}
}
