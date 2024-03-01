
module test (a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15, 
   b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15, sum );
   input a_0;
  input a_1;
  input a_2;
  input a_3;
  input a_4;
  input a_5;
  input a_6;
  input a_7;
  input a_8;
  input a_9;
  input a_10;
  input a_11;
  input a_12;
  input a_13;
  input a_14;
  input a_15;

  input b_0;
  input b_1;
  input b_2;
  input b_3;
  input b_4;
  input b_5;
  input b_6;
  input b_7;
  input b_8;
  input b_9;
  input b_10;
  input b_11;
  input b_12;
  input b_13;
  input b_14;
  input b_15;
  output sum;

	not gate_sum (sum, n3);
	nand gate_n3 (n3, n4, n5, n6, n7);
	nor gate_n7 (n7, n8, n9, n10, n11);
	xor gate_n11 (n11, b_12, a_12);
	xor gate_n10 (n10, b_11, a_11);
	xor gate_n9 (n9, b_10, a_10);
	xor gate_n8 (n8, b_0, a_0);
	nor gate_n6 (n6, n12, n13, n14, n15);
	xor gate_n15 (n15, b_1, a_1);
	xor gate_n14 (n14, b_15, a_15);
	xor gate_n13 (n13, b_14, a_14);
	xor gate_n12 (n12, b_13, a_13);
	nor gate_n5 (n5, n16, n17, n18, n19);
	xor gate_n19 (n19, b_5, a_5);
	xor gate_n18 (n18, b_4, a_4);
	xor gate_n17 (n17, b_3, a_3);
	xor gate_n16 (n16, b_2, a_2);
	nor gate_n4 (n4, n20, n21, n22, n23);
	xor gate_n23 (n23, b_9, a_9);
	xor gate_n22 (n22, b_8, a_8);
	xor gate_n21 (n21, b_7, a_7);
	xor gate_n20 (n20, b_6, a_6);
endmodule

