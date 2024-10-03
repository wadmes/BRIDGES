
module test ( a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, sum );
  
  input a_0;
  input a_1;
  input a_2;
  input a_3;
  input a_4;
  input a_5;
  input a_6;
  input a_7;
  
  input b_0;
  input b_1;
  input b_2;
  input b_3;
  input b_4;
  input b_5;
  input b_6;
  input b_7;
  
  output sum;
  

	and gate_sum (sum, n3, n4);
	nor gate_n4 (n4, n5, n6, n7, n8);
	xor gate_n8 (n8, b_5, a_5);
	xor gate_n7 (n7, b_4, a_4);
	xor gate_n6 (n6, b_7, a_7);
	xor gate_n5 (n5, b_3, a_3);
	nor gate_n3 (n3, n9, n10, n11, n12);
	xor gate_n12 (n12, b_2, a_2);
	xor gate_n11 (n11, b_1, a_1);
	xor gate_n10 (n10, b_0, a_0);
	xor gate_n9 (n9, b_6, a_6);
endmodule

