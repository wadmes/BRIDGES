
module test (cin,  a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12, a_13, a_14, a_15, 
   b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15,
   anoymous_0, anoymous_1, anoymous_2, anoymous_3, anoymous_4, anoymous_5, anoymous_6, anoymous_7, anoymous_8, anoymous_9, anoymous_10, anoymous_11, 
   anoymous_12, anoymous_13, anoymous_14, anoymous_15, cout );
  
  input cin;
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


  output anoymous_0;
  output anoymous_1;
  output anoymous_2;
  output anoymous_3;
  output anoymous_4;
  output anoymous_5;
  output anoymous_6;
  output anoymous_7;
  output anoymous_8;
  output anoymous_9;
  output anoymous_10;
  output anoymous_11;
  output anoymous_12;
  output anoymous_13;
  output anoymous_14;
  output anoymous_15;
  
  output cout;

  //correct key = 101110101000010


	xor gate_anoymous_9 (anoymous_9, n64, n65);
	xor gate_anoymous_8 (anoymous_8, n66, n67);
	xor gate_anoymous_7 (anoymous_7, n68, n69);
	xor gate_anoymous_6 (anoymous_6, n70, n71);
	xor gate_anoymous_5 (anoymous_5, n72, n73);
	xor gate_anoymous_4 (anoymous_4, n74, n75);
	xor gate_anoymous_3 (anoymous_3, n76, n77);
	xor gate_anoymous_2 (anoymous_2, n78, n79);
	xor gate_anoymous_1 (anoymous_1, n80, n81);
	xor gate_anoymous_15 (anoymous_15, n82, n83);
	xor gate_anoymous_14 (anoymous_14, n84, n85);
	xor gate_anoymous_13 (anoymous_13, n86, n87);
	xor gate_anoymous_12 (anoymous_12, n88, n89);
	xor gate_anoymous_11 (anoymous_11, n90, n91);
	xor gate_anoymous_10 (anoymous_10, n92, n93);
	xor gate_anoymous_0 (anoymous_0, cin, n94);
	nand gate_cout (cout, n95, n96);
	nand gate_n96 (n96, n82, n83);
	nand gate_n83 (n83, n97, n98);
	nand gate_n98 (n98, n84, n85);
	or gate_n85 (n85, n99, n100);
	and gate_n100 (n100, a_13, b_13);
	and gate_n99 (n99, n87, n86);
	xor gate_n86 (n86, b_13, a_13);
	nand gate_n87 (n87, n101, n102);
	nand gate_n102 (n102, n88, n89);
	or gate_n89 (n89, n103, n104);
	and gate_n104 (n104, a_11, b_11);
	and gate_n103 (n103, n91, n90);
	xor gate_n90 (n90, b_11, a_11);
	nand gate_n91 (n91, n105, n106);
	nand gate_n106 (n106, n92, n93);
	or gate_n93 (n93, n107, n108);
	and gate_n108 (n108, a_9, b_9);
	and gate_n107 (n107, n65, n64);
	xor gate_n64 (n64, b_9, a_9);
	nand gate_n65 (n65, n109, n110);
	nand gate_n110 (n110, n66, n67);
	or gate_n67 (n67, n111, n112);
	and gate_n112 (n112, a_7, b_7);
	and gate_n111 (n111, n69, n68);
	xor gate_n68 (n68, b_7, a_7);
	nand gate_n69 (n69, n113, n114);
	nand gate_n114 (n114, n70, n71);
	nand gate_n71 (n71, n115, n116);
	nand gate_n116 (n116, n72, n73);
	nand gate_n73 (n73, n117, n118);
	nand gate_n118 (n118, n74, n75);
	nand gate_n75 (n75, n119, n120);
	nand gate_n120 (n120, n76, n77);
	nand gate_n77 (n77, n121, n122);
	nand gate_n122 (n122, n78, n79);
	nand gate_n79 (n79, n123, n124);
	nand gate_n124 (n124, n80, n81);
	nand gate_n81 (n81, n125, n126);
	nand gate_n126 (n126, cin, n94);
	xor gate_n94 (n94, b_0, a_0);
	nand gate_n125 (n125, a_0, b_0);
	xor gate_n80 (n80, b_1, a_1);
	nand gate_n123 (n123, a_1, b_1);
	xor gate_n78 (n78, b_2, a_2);
	nand gate_n121 (n121, a_2, b_2);
	xor gate_n76 (n76, b_3, a_3);
	nand gate_n119 (n119, a_3, b_3);
	xor gate_n74 (n74, b_4, a_4);
	nand gate_n117 (n117, a_4, b_4);
	xor gate_n72 (n72, b_5, a_5);
	nand gate_n115 (n115, a_5, b_5);
	xor gate_n70 (n70, b_6, a_6);
	nand gate_n113 (n113, a_6, b_6);
	xor gate_n66 (n66, b_8, a_8);
	nand gate_n109 (n109, a_8, b_8);
	xor gate_n92 (n92, b_10, a_10);
	nand gate_n105 (n105, a_10, b_10);
	xor gate_n88 (n88, b_12, a_12);
	nand gate_n101 (n101, a_12, b_12);
	xor gate_n84 (n84, b_14, a_14);
	nand gate_n97 (n97, a_14, b_14);
	xor gate_n82 (n82, b_15, a_15);
	nand gate_n95 (n95, a_15, b_15);
endmodule

