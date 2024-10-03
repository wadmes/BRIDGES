
module adder_4_bit (cin, a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, sum_0, sum_1, sum_2, sum_3, cout );

  input cin;

  input a_0;
  input a_1;
  input a_2;
  input a_3;

  input b_0;
  input b_1;
  input b_2;
  input b_3;

  output sum_0;
  output sum_1;
  output sum_2;
  output sum_3;

  output cout;

	xor gate_sum_3 (sum_3, n16, n17);
	xor gate_sum_2 (sum_2, n18, n19);
	xor gate_sum_1 (sum_1, n20, n21);
	xor gate_sum_0 (sum_0, cin, n22);
	nand gate_cout (cout, n23, n24);
	nand gate_n24 (n24, n16, n17);
	nand gate_n17 (n17, n25, n26);
	nand gate_n26 (n26, n18, n19);
	nand gate_n19 (n19, n27, n28);
	nand gate_n28 (n28, n20, n21);
	nand gate_n21 (n21, n29, n30);
	nand gate_n30 (n30, cin, n22);
	xor gate_n22 (n22, b_0, a_0);
	nand gate_n29 (n29, a_0, b_0);
	xor gate_n20 (n20, b_1, a_1);
	nand gate_n27 (n27, a_1, b_1);
	xor gate_n18 (n18, b_2, a_2);
	nand gate_n25 (n25, a_2, b_2);
	xor gate_n16 (n16, b_3, a_3);
	nand gate_n23 (n23, a_3, b_3);

endmodule

