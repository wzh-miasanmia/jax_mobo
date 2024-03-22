from .base import ReconstructionStencil, epsilon_

coef_smoothness_0_01_ = 22658.0
coef_smoothness_0_02_ = -208501.0
coef_smoothness_0_03_ = 364863.0
coef_smoothness_0_04_ = -288007.0
coef_smoothness_0_05_ = 86329.0
coef_smoothness_0_06_ = 482963.0
coef_smoothness_0_07_ = -1704396.0
coef_smoothness_0_08_ = 1358458.0
coef_smoothness_0_09_ = -411487.0
coef_smoothness_0_10_ = 1521393.0
coef_smoothness_0_11_ = -2462076.0
coef_smoothness_0_12_ = 758823.0
coef_smoothness_0_13_ = 1020563.0
coef_smoothness_0_14_ = -649501.0
coef_smoothness_0_15_ = 107918.0

coef_smoothness_1_01_ = 6908.0
coef_smoothness_1_02_ = -60871.0
coef_smoothness_1_03_ = 99213.0
coef_smoothness_1_04_ = -70237.0
coef_smoothness_1_05_ = 18079.0
coef_smoothness_1_06_ = 138563.0
coef_smoothness_1_07_ = -464976.0
coef_smoothness_1_08_ = 337018.0
coef_smoothness_1_09_ = -88297.0
coef_smoothness_1_10_ = 406293.0
coef_smoothness_1_11_ = -611976.0
coef_smoothness_1_12_ = 165153.0
coef_smoothness_1_13_ = 242723.0
coef_smoothness_1_14_ = -140251.0
coef_smoothness_1_15_ = 22658.0

coef_smoothness_2_01_ = 6908.0
coef_smoothness_2_02_ = -51001.0
coef_smoothness_2_03_ = 67923.0
coef_smoothness_2_04_ = -38947.0
coef_smoothness_2_05_ = 8209.0
coef_smoothness_2_06_ = 104963.0
coef_smoothness_2_07_ = -299076.0
coef_smoothness_2_08_ = 179098.0
coef_smoothness_2_09_ = -38947.0
coef_smoothness_2_10_ = 231153.0
coef_smoothness_2_11_ = -299076.0
coef_smoothness_2_12_ = 67923.0
coef_smoothness_2_13_ = 104963.0
coef_smoothness_2_14_ = -51001.0
coef_smoothness_2_15_ = 6908.0

coef_smoothness_3_01_ = 22658.0
coef_smoothness_3_02_ = -140251.0
coef_smoothness_3_03_ = 165153.0
coef_smoothness_3_04_ = -88297.0
coef_smoothness_3_05_ = 18079.0
coef_smoothness_3_06_ = 242723.0
coef_smoothness_3_07_ = -611976.0
coef_smoothness_3_08_ = 337018.0
coef_smoothness_3_09_ = -70237.0
coef_smoothness_3_10_ = 406293.0
coef_smoothness_3_11_ = -464976.0
coef_smoothness_3_12_ = 99213.0
coef_smoothness_3_13_ = 138563.0
coef_smoothness_3_14_ = -60871.0
coef_smoothness_3_15_ = 6908.0

coef_smoothness_4_01_ = 107918.0
coef_smoothness_4_02_ = -649501.0
coef_smoothness_4_03_ = 758823.0
coef_smoothness_4_04_ = -411487.0
coef_smoothness_4_05_ = 86329.0
coef_smoothness_4_06_ = 1020563.0
coef_smoothness_4_07_ = -2462076.0
coef_smoothness_4_08_ = 1358458.0
coef_smoothness_4_09_ = -288007.0
coef_smoothness_4_10_ = 1521393.0
coef_smoothness_4_11_ = -1704396.0
coef_smoothness_4_12_ = 364863.0
coef_smoothness_4_13_ = 482963.0
coef_smoothness_4_14_ = -208501.0
coef_smoothness_4_15_ = 22658.0

coef_weights_1_ = 1.0 / 126.0
coef_weights_2_ = 10.0 / 63.0
coef_weights_3_ = 10.0 / 21.0
coef_weights_4_ = 20.0 / 63.0
coef_weights_5_ = 5.0 / 126.0

coef_stencils_1_ = 12.0 / 60.0
coef_stencils_2_ = -63.0 / 60.0
coef_stencils_3_ = 137.0 / 60.0
coef_stencils_4_ = -163.0 / 60.0
coef_stencils_5_ = 137.0 / 60.0

coef_stencils_6_  = -3.0 / 60.0
coef_stencils_7_  = 17.0 / 60.0
coef_stencils_8_  = -43.0 / 60.0
coef_stencils_9_  = 77.0 / 60.0
coef_stencils_10_ = 12.0 / 60.0

coef_stencils_11_ = 2.0 / 60.0
coef_stencils_12_ = -13.0 / 60.0
coef_stencils_13_ = 47.0 / 60.0
coef_stencils_14_ = 27.0 / 60.0
coef_stencils_15_ = -3.0 / 60.0

coef_stencils_16_ = -3.0 / 60.0
coef_stencils_17_ = 27.0 / 60.0
coef_stencils_18_ = 47.0 / 60.0
coef_stencils_19_ = -13.0 / 60.0
coef_stencils_20_ = 2.0 / 60.0

coef_stencils_21_ = 12.0 / 60.0
coef_stencils_22_ = 77.0 / 60.0
coef_stencils_23_ = -43.0 / 60.0
coef_stencils_24_ = 17.0 / 60.0
coef_stencils_25_ = -3.0 / 60.0

epsilon_weno9_ = 1.0e-10


class WENO9(ReconstructionStencil):
    def __init__(self, nonlinear=True):
        self.nonlinear = nonlinear

    def apply(self, value):
        if not isinstance(value, list):
            raise Exception("Inputs must be a list.")
        elif len(value) != 9:
            raise Exception("Inputs must have at least 9 values.")
        else:
            v1 = value[0]
            v2 = value[1]
            v3 = value[2]
            v4 = value[3]
            v5 = value[4]
            v6 = value[5]
            v7 = value[6]
            v8 = value[7]
            v9 = value[8]

            if self.nonlinear:

                s11 = coef_smoothness_0_01_ * v1 + coef_smoothness_0_02_ * v2 + coef_smoothness_0_03_ * v3 + coef_smoothness_0_04_ * v4 + coef_smoothness_0_05_ * v5
                s12 = coef_smoothness_0_06_ * v2 + coef_smoothness_0_07_ * v3 + coef_smoothness_0_08_ * v4 + coef_smoothness_0_09_ * v5
                s13 = coef_smoothness_0_10_ * v3 + coef_smoothness_0_11_ * v4 + coef_smoothness_0_12_ * v5
                s14 = coef_smoothness_0_13_ * v4 + coef_smoothness_0_14_ * v5
                s15 = coef_smoothness_0_15_ * v5

                s1 = v1 * s11 + v2 * s12 + v3 * s13 + v4 * s14 + v5 * s15

                s21 = coef_smoothness_1_01_ * v2 + coef_smoothness_1_02_ * v3 + coef_smoothness_1_03_ * v4 + coef_smoothness_1_04_ * v5 + coef_smoothness_1_05_ * v6
                s22 = coef_smoothness_1_06_ * v3 + coef_smoothness_1_07_ * v4 + coef_smoothness_1_08_ * v5 + coef_smoothness_1_09_ * v6
                s23 = coef_smoothness_1_10_ * v4 + coef_smoothness_1_11_ * v5 + coef_smoothness_1_12_ * v6
                s24 = coef_smoothness_1_13_ * v5 + coef_smoothness_1_14_ * v6
                s25 = coef_smoothness_1_15_ * v6

                s2 = v2 * s21 + v3 * s22 + v4 * s23 + v5 * s24 + v6 * s25

                s31 = coef_smoothness_2_01_ * v3 + coef_smoothness_2_02_ * v4 + coef_smoothness_2_03_ * v5 + coef_smoothness_2_04_ * v6 + coef_smoothness_2_05_ * v7
                s32 = coef_smoothness_2_06_ * v4 + coef_smoothness_2_07_ * v5 + coef_smoothness_2_08_ * v6 + coef_smoothness_2_09_ * v7
                s33 = coef_smoothness_2_10_ * v5 + coef_smoothness_2_11_ * v6 + coef_smoothness_2_12_ * v7
                s34 = coef_smoothness_2_13_ * v6 + coef_smoothness_2_14_ * v7
                s35 = coef_smoothness_2_15_ * v7

                s3 = v3 * s31 + v4 * s32 + v5 * s33 + v6 * s34 + v7 * s35

                s41 = coef_smoothness_3_01_ * v4 + coef_smoothness_3_02_ * v5 + coef_smoothness_3_03_ * v6 + coef_smoothness_3_04_ * v7 + coef_smoothness_3_05_ * v8
                s42 = coef_smoothness_3_06_ * v5 + coef_smoothness_3_07_ * v6 + coef_smoothness_3_08_ * v7 + coef_smoothness_3_09_ * v8
                s43 = coef_smoothness_3_10_ * v6 + coef_smoothness_3_11_ * v7 + coef_smoothness_3_12_ * v8
                s44 = coef_smoothness_3_13_ * v7 + coef_smoothness_3_14_ * v8
                s45 = coef_smoothness_3_15_ * v8

                s4 = v4 * s41 + v5 * s42 + v6 * s43 + v7 * s44 + v8 * s45

                s51 = coef_smoothness_4_01_ * v5 + coef_smoothness_4_02_ * v6 + coef_smoothness_4_03_ * v7 + coef_smoothness_4_04_ * v8 + coef_smoothness_4_05_ * v9
                s52 = coef_smoothness_4_06_ * v6 + coef_smoothness_4_07_ * v7 + coef_smoothness_4_08_ * v8 + coef_smoothness_4_09_ * v9
                s53 = coef_smoothness_4_10_ * v7 + coef_smoothness_4_11_ * v8 + coef_smoothness_4_12_ * v9
                s54 = coef_smoothness_4_13_ * v8 + coef_smoothness_4_14_ * v9
                s55 = coef_smoothness_4_15_ * v9

                s5 = v5 * s51 + v6 * s52 + v7 * s53 + v8 * s54 + v9 * s55

                a1 = coef_weights_1_ / ( ( s1 + epsilon_weno9_ ) * ( s1 + epsilon_weno9_ ) )
                a2 = coef_weights_2_ / ( ( s2 + epsilon_weno9_ ) * ( s2 + epsilon_weno9_ ) )
                a3 = coef_weights_3_ / ( ( s3 + epsilon_weno9_ ) * ( s3 + epsilon_weno9_ ) )
                a4 = coef_weights_4_ / ( ( s4 + epsilon_weno9_ ) * ( s4 + epsilon_weno9_ ) )
                a5 = coef_weights_5_ / ( ( s5 + epsilon_weno9_ ) * ( s5 + epsilon_weno9_ ) )

                one_a_sum = 1.0 / ( a1 + a2 + a3 + a4 + a5 )

                w1 = a1 * one_a_sum
                w2 = a2 * one_a_sum
                w3 = a3 * one_a_sum
                w4 = a4 * one_a_sum
                w5 = a5 * one_a_sum
            else:
                w1 = coef_weights_1_
                w2 = coef_weights_2_
                w3 = coef_weights_3_
                w4 = coef_weights_4_
                w5 = coef_weights_5_

            return w1 * ( coef_stencils_1_ * v1 + coef_stencils_2_ * v2 + coef_stencils_3_ * v3 + coef_stencils_4_ * v4 + coef_stencils_5_ * v5 ) + w2 * ( coef_stencils_6_ * v2 + coef_stencils_7_ * v3 + coef_stencils_8_ * v4 + coef_stencils_9_ * v5 + coef_stencils_10_ * v6 ) + w3 * ( coef_stencils_11_ * v3 + coef_stencils_12_ * v4 + coef_stencils_13_ * v5 + coef_stencils_14_ * v6 + coef_stencils_15_ * v7 ) + w4 * ( coef_stencils_16_ * v4 + coef_stencils_17_ * v5 + coef_stencils_18_ * v6 + coef_stencils_19_ * v7 + coef_stencils_20_ * v8 ) + w5 * ( coef_stencils_21_ * v5 + coef_stencils_22_ * v6 + coef_stencils_23_ * v7 + coef_stencils_24_ * v8 + coef_stencils_25_ * v9 )
