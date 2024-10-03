
module comparator_4_bit (a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, sum );
  input a_0;
  input a_1;
  input a_2;
  input a_3;
  input b_0;
  input b_1;
  input b_2;
  input b_3;
  output sum;

	nor gate_sum (sum, n5, n6, n7, n8);
	xor gate_n8 (n8, b_1, a_1);
	xor gate_n7 (n7, b_2, a_2);
	xor gate_n6 (n6, b_3, a_3);
	xor gate_n5 (n5, b_0, a_0);
endmodule

