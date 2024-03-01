
module test ( cin, a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, 
  b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, sum_0, sum_1, sum_2, sum_3, sum_4, sum_5, sum_6, sum_7, cout );

  input cin;
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

  output sum_0;
  output sum_1;
  output sum_2;
  output sum_3;
  output sum_4;
  output sum_5;
  output sum_6;
  output sum_7;
 
  output cout;

	xor gate_sum_7 (sum_7, n32, n33);
	xor gate_sum_6 (sum_6, n34, n35);
	xor gate_sum_5 (sum_5, n36, n37);
	xor gate_sum_4 (sum_4, n38, n39);
	xor gate_sum_3 (sum_3, n40, n41);
	xor gate_sum_2 (sum_2, n42, n43);
	xor gate_sum_1 (sum_1, n44, n45);
	xor gate_sum_0 (sum_0, cin, n46);
	nand gate_cout (cout, n47, n48);
	nand gate_n48 (n48, n32, n33);
	nand gate_n33 (n33, n49, n50);
	nand gate_n50 (n50, n34, n35);
	nand gate_n35 (n35, n51, n52);
	nand gate_n52 (n52, n36, n37);
	nand gate_n37 (n37, n53, n54);
	nand gate_n54 (n54, n38, n39);
	nand gate_n39 (n39, n55, n56);
	nand gate_n56 (n56, n40, n41);
	nand gate_n41 (n41, n57, n58);
	nand gate_n58 (n58, n42, n43);
	nand gate_n43 (n43, n59, n60);
	nand gate_n60 (n60, n44, n45);
	nand gate_n45 (n45, n61, n62);
	nand gate_n62 (n62, cin, n46);
	xor gate_n46 (n46, b_0, a_0);
	nand gate_n61 (n61, a_0, b_0);
	xor gate_n44 (n44, b_1, a_1);
	nand gate_n59 (n59, a_1, b_1);
	xor gate_n42 (n42, b_2, a_2);
	nand gate_n57 (n57, a_2, b_2);
	xor gate_n40 (n40, b_3, a_3);
	nand gate_n55 (n55, a_3, b_3);
	xor gate_n38 (n38, b_4, a_4);
	nand gate_n53 (n53, a_4, b_4);
	xor gate_n36 (n36, b_5, a_5);
	nand gate_n51 (n51, a_5, b_5);
	xor gate_n34 (n34, b_6, a_6);
	nand gate_n49 (n49, a_6, b_6);
	xor gate_n32 (n32, b_7, a_7);
	nand gate_n47 (n47, a_7, b_7);
endmodule

