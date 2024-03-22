from .base import ReconstructionStencil, epsilon_

coef_smoothness_1_ = 13.0 / 12.0
coef_smoothness_2_ = 0.25

coef_smoothness_11_ = 1.0
coef_smoothness_12_ = -2.0
coef_smoothness_13_ = 1.0
coef_smoothness_14_ = 1.0
coef_smoothness_15_ = -1.0

coef_smoothness_21_ = 1.0
coef_smoothness_22_ = -2.0
coef_smoothness_23_ = 1.0
coef_smoothness_24_ = 3.0
coef_smoothness_25_ = -4.0
coef_smoothness_26_ = 1.0

coef_smoothness_31_ = 1.0
coef_smoothness_32_ = -2.0
coef_smoothness_33_ = 1.0
coef_smoothness_34_ = 1.0
coef_smoothness_35_ = -4.0
coef_smoothness_36_ = 3.0

coef_stencils_1_ = -1.0
coef_stencils_2_ = 5.0
coef_stencils_3_ = 2.0
coef_stencils_4_ = 2.0
coef_stencils_5_ = 5.0
coef_stencils_6_ = -1.0
coef_stencils_7_ = 2.0
coef_stencils_8_ = -7.0
coef_stencils_9_ = 11.0

multiplyer_stencils_ = 1.0 / 6.0


class TENO5(ReconstructionStencil):
    def __init__(
            self,
            d0=0.55,
            d1=0.40,
            d2=0.05,
            CT=1.0e-5,
            Cq=1,
            q=6,
            nonlinear=True
    ):
        self.d0, self.d1, self.d2 = d0, d1, d2
        self.CT = CT
        self.Cq, self.q = Cq, q
        # print(self.d0, self.d1, self.d2, self.CT, self.Cq, self.q)
        self.nonlinear = nonlinear

    def apply(self, value):
        if not isinstance(value, list):
            raise Exception("Inputs must be a list.")
        elif len(value) != 5:
            raise Exception("Inputs must have at least 5 values.")
        else:
            v1 = value[0]
            v2 = value[1]
            v3 = value[2]
            v4 = value[3]
            v5 = value[4]

            Variation1 = coef_stencils_1_ * v2 + coef_stencils_2_ * v3 + coef_stencils_3_ * v4
            Variation2 = coef_stencils_4_ * v3 + coef_stencils_5_ * v4 + coef_stencils_6_ * v5
            Variation3 = coef_stencils_7_ * v1 + coef_stencils_8_ * v2 + coef_stencils_9_ * v3

            if self.nonlinear:
                s11 = coef_smoothness_11_ * v2 + coef_smoothness_12_ * v3 + coef_smoothness_13_ * v4
                s12 = coef_smoothness_14_ * v2 + coef_smoothness_15_ * v4
                s1 = coef_smoothness_1_ * s11 * s11 + coef_smoothness_2_ * s12 * s12 # beta_0

                s21 = coef_smoothness_21_ * v3 + coef_smoothness_22_ * v4 + coef_smoothness_23_ * v5
                s22 = coef_smoothness_24_ * v3 + coef_smoothness_25_ * v4 + coef_smoothness_26_ * v5
                s2 = coef_smoothness_1_ * s21 * s21 + coef_smoothness_2_ * s22 * s22 # beta_1

                s31 = coef_smoothness_31_ * v1 + coef_smoothness_32_ * v2 + coef_smoothness_33_ * v3
                s32 = coef_smoothness_34_ * v1 + coef_smoothness_35_ * v2 + coef_smoothness_36_ * v3
                s3 = coef_smoothness_1_ * s31 * s31 + coef_smoothness_2_ * s32 * s32 # beta_2

                tau5 = abs(s3 - s2)

                a1 = self.Cq + tau5 / (s1 + epsilon_)
                a2 = self.Cq + tau5 / (s2 + epsilon_)
                a3 = self.Cq + tau5 / (s3 + epsilon_)

                a1_temp = a1
                a2_temp = a2
                a3_temp = a3

                for i in range(self.q - 1):
                    a1 *= a1_temp
                    a2 *= a2_temp
                    a3 *= a3_temp

                one_a_sum = 1.0 / (a1 + a2 + a3)

                b1 = 0.0 if a1 * one_a_sum < self.CT else 1.0
                b2 = 0.0 if a2 * one_a_sum < self.CT else 1.0
                b3 = 0.0 if a3 * one_a_sum < self.CT else 1.0



                w1 = self.d0 * b1
                w2 = self.d1 * b2
                w3 = self.d2 * b3

                one_w_sum = 1.0 / (w1 + w2 + w3)

                w1 = w1 * one_w_sum
                w2 = w2 * one_w_sum
                w3 = w3 * one_w_sum
            else:
                w1 = self.d0
                w2 = self.d1
                w3 = self.d2

            return (w1 * Variation1 + w2 * Variation2 + w3 * Variation3) * multiplyer_stencils_

    def apply_with_weights(self, value):
        if not isinstance(value, list):
            raise Exception("Inputs must be a list.")
        elif len(value) != 5:
            raise Exception("Inputs must have at least 5 values.")
        else:
            v1 = value[0]
            v2 = value[1]
            v3 = value[2]
            v4 = value[3]
            v5 = value[4]

            s11 = coef_smoothness_11_ * v2 + coef_smoothness_12_ * v3 + coef_smoothness_13_ * v4
            s12 = coef_smoothness_14_ * v2 + coef_smoothness_15_ * v4
            s1 = coef_smoothness_1_ * s11 * s11 + coef_smoothness_2_ * s12 * s12

            s21 = coef_smoothness_21_ * v3 + coef_smoothness_22_ * v4 + coef_smoothness_23_ * v5
            s22 = coef_smoothness_24_ * v3 + coef_smoothness_25_ * v4 + coef_smoothness_26_ * v5
            s2 = coef_smoothness_1_ * s21 * s21 + coef_smoothness_2_ * s22 * s22

            s31 = coef_smoothness_31_ * v1 + coef_smoothness_32_ * v2 + coef_smoothness_33_ * v3
            s32 = coef_smoothness_34_ * v1 + coef_smoothness_35_ * v2 + coef_smoothness_36_ * v3
            s3 = coef_smoothness_1_ * s31 * s31 + coef_smoothness_2_ * s32 * s32

            tau5 = abs(s3 - s2)

            a1 = (self.Cq + tau5 / (s1 + epsilon_))**self.q
            a2 = (self.Cq + tau5 / (s2 + epsilon_))**self.q
            a3 = (self.Cq + tau5 / (s3 + epsilon_))**self.q

            # a1_temp = a1
            # a2_temp = a2
            # a3_temp = a3
            #
            # for i in range(self.q - 1):
            #     a1 *= a1_temp
            #     a2 *= a2_temp
            #     a3 *= a3_temp

            one_a_sum = 1.0 / (a1 + a2 + a3)

            b1 = 0.0 if a1 * one_a_sum < self.CT else 1.0
            b2 = 0.0 if a2 * one_a_sum < self.CT else 1.0
            b3 = 0.0 if a3 * one_a_sum < self.CT else 1.0

            Variation1 = coef_stencils_1_ * v2 + coef_stencils_2_ * v3 + coef_stencils_3_ * v4
            Variation2 = coef_stencils_4_ * v3 + coef_stencils_5_ * v4 + coef_stencils_6_ * v5
            Variation3 = coef_stencils_7_ * v1 + coef_stencils_8_ * v2 + coef_stencils_9_ * v3

            w1 = self.d0 * b1
            w2 = self.d1 * b2
            w3 = self.d2 * b3

            one_w_sum = 1.0 / (w1 + w2 + w3)

            w1_normalized = w1 * one_w_sum
            w2_normalized = w2 * one_w_sum
            w3_normalized = w3 * one_w_sum

            constructed = (w1_normalized * Variation1 + w2_normalized * Variation2 + w3_normalized * Variation3)
            constructed *= multiplyer_stencils_

            return constructed, w2_normalized

    def smoothness_indicator_weno5(self, value):

        if not isinstance(value, list):
            raise Exception("Inputs must be a list.")
        elif len(value) != 5:
            raise Exception("Inputs must have at least 5 values.")
        else:
            v1 = value[0]
            v2 = value[1]
            v3 = value[2]
            v4 = value[3]
            v5 = value[4]

            s11 = coef_smoothness_11_ * v2 + coef_smoothness_12_ * v3 + coef_smoothness_13_ * v4
            s12 = coef_smoothness_14_ * v2 + coef_smoothness_15_ * v4
            s1 = coef_smoothness_1_ * s11 * s11 + coef_smoothness_2_ * s12 * s12

            s21 = coef_smoothness_21_ * v3 + coef_smoothness_22_ * v4 + coef_smoothness_23_ * v5
            s22 = coef_smoothness_24_ * v3 + coef_smoothness_25_ * v4 + coef_smoothness_26_ * v5
            s2 = coef_smoothness_1_ * s21 * s21 + coef_smoothness_2_ * s22 * s22

            s31 = coef_smoothness_31_ * v1 + coef_smoothness_32_ * v2 + coef_smoothness_33_ * v3
            s32 = coef_smoothness_34_ * v1 + coef_smoothness_35_ * v2 + coef_smoothness_36_ * v3
            s3 = coef_smoothness_1_ * s31 * s31 + coef_smoothness_2_ * s32 * s32

            epsilon_weno5_ = 1.0e-6
            one_s1 = 1.0 / ((s1 + epsilon_weno5_) * (s1 + epsilon_weno5_))
            one_s2 = 1.0 / ((s2 + epsilon_weno5_) * (s2 + epsilon_weno5_))
            one_s3 = 1.0 / ((s3 + epsilon_weno5_) * (s3 + epsilon_weno5_))

            a1_weno5 = one_s1 / (one_s1 + one_s2 + one_s3)
            a2_weno5 = one_s2 / (one_s1 + one_s2 + one_s3)
            a3_weno5 = one_s3 / (one_s1 + one_s2 + one_s3)

            return a1_weno5, a2_weno5, a3_weno5

    def get_min_indicator(self, value):
        indicator = self.smoothness_indicator_weno5(value)
        return min(indicator)