
module test ( cin, a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, 
  b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, anoymous_0, anoymous_1, anoymous_2, anoymous_3, anoymous_4, anoymous_5, anoymous_6, anoymous_7, cout );

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

  output anoymous_0;
  output anoymous_1;
  output anoymous_2;
  output anoymous_3;
  output anoymous_4;
  output anoymous_5;
  output anoymous_6;
  output anoymous_7;
 
  output cout;

	xor gate_anoymous_7 (anoymous_7, n32, n33);
	xor gate_anoymous_6 (anoymous_6, n34, n35);
	xor gate_anoymous_5 (anoymous_5, n36, n37);
	xor gate_anoymous_4 (anoymous_4, n38, n39);
	xor gate_anoymous_3 (anoymous_3, n40, n41);
	xor gate_anoymous_2 (anoymous_2, n42, n43);
	xor gate_anoymous_1 (anoymous_1, n44, n45);
	xor gate_anoymous_0 (anoymous_0, cin, n46);
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

