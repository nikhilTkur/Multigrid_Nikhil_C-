# Multigrid_Nikhil_C-
This repo consists of the C++code for Multigrid (includes SYCL and oneMKL)

						4-LEVEL-FMG CYCLE WITH V-CYCLE
  Levels

	1	 \																	                                           / V=>S\								                     /=>S=>FINAL SOLUTION
																			                                           I		     R								                 I
	2	   \							                     / V=>S\					               /=>S/		        \=>S\					              /=>S/
										                      I		    R				              	I					              R				             I
	3		   \		      / V=>S\		      / =>S/		      \=>S\		       / =>S/					                 \=>S\		     /=>S/
					         I		    R	     I					            R		    I								                       R		  I
	4		      \_ DS_/		       \_DS_/					                \_DS_/								                        \_DS_/
	      
	DS => DIRECT SOLVER (EIGEN SPARSELU)
	I => INTERPOLATION
	R => RESTRICTION
	S => SMOOTHER (JACOBI)	
