
coef_stencils_1_ = -1.0
coef_stencils_2_ = 5.0
coef_stencils_3_ = 2.0
coef_stencils_4_ = 2.0
coef_stencils_5_ = 5.0
coef_stencils_6_ = -1.0
coef_stencils_7_ = 2.0
coef_stencils_8_ = -7.0
coef_stencils_9_ = 11.0

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


class TENO5SENSOR:
    def __init__(
        self,
        coefficients: list
    ):
        self.eno_cutoff = coefficients[0]
        self.h_cutoff = 10 ** coefficients[1]
        self.h_c = coefficients[2]
        self.h_q = coefficients[3]
        self.h_eta = coefficients[4]
        self.l_eta = coefficients[5]
        

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

        s11 = coef_smoothness_11_ * v2 + coef_smoothness_12_ * v3 + coef_smoothness_13_ * v4
        s12 = coef_smoothness_14_ * v2 + coef_smoothness_15_ * v4
        s1 = coef_smoothness_1_ * s11 * s11 + coef_smoothness_2_ * s12 * s12

        s21 = coef_smoothness_21_ * v3 + coef_smoothness_22_ * v4 + coef_smoothness_23_ * v5
        s22 = coef_smoothness_24_ * v3 + coef_smoothness_25_ * v4 + coef_smoothness_26_ * v5
        s2 = coef_smoothness_1_ * s21 * s21 + coef_smoothness_2_ * s22 * s22

        s31 = coef_smoothness_31_ * v1 + coef_smoothness_32_ * v2 + coef_smoothness_33_ * v3
        s32 = coef_smoothness_34_ * v1 + coef_smoothness_35_ * v2 + coef_smoothness_36_ * v3
        s3 = coef_smoothness_1_ * s31 * s31 + coef_smoothness_2_ * s32 * s32
        
        Variation1 = coef_stencils_1_ * v2 + coef_stencils_2_ * v3 + coef_stencils_3_ * v4
        Variation2 = coef_stencils_4_ * v3 + coef_stencils_5_ * v4 + coef_stencils_6_ * v5
        Variation3 = coef_stencils_7_ * v1 + coef_stencils_8_ * v2 + coef_stencils_9_ * v3
        
        epsilon_ = 1.0e-30
        tau5 = abs( s3 - s2 )
        CT_ = 1e-5
        
        si1 = (1.0 / (s1 + 1.0e-4))**6
        si2 = (1.0 / (s2 + 1.0e-4))**6
        si3 = (1.0 / (s3 + 1.0e-4))**6
        one_si_sum = 1.0 / (si1 + si2 + si3)
        si_mininum = min(si1 * one_si_sum, si2 * one_si_sum, si3 * one_si_sum)
        # self.h_cutoff = 
        # print(10**self.h_cutoff)
        if si_mininum < self.eno_cutoff:
            a1 = (1.0 + tau5 / ( s1 + epsilon_ ))**6
            a2 = (1.0 + tau5 / ( s2 + epsilon_ ))**6
            a3 = (1.0 + tau5 / ( s3 + epsilon_ ))**6
            one_a_sum = 1.0 / ( a1 + a2 + a3 )
            
            b1 = 0 if a1 * one_a_sum < CT_ else 1.0
            b2 = 0 if a2 * one_a_sum < CT_ else 1.0
            b3 = 0 if a3 * one_a_sum < CT_ else 1.0
            
            eta = 0.4
            w1 = (2 + eta) / 4 * b1
            w2 = (1 - eta) / 2 * b2
            w3 = eta / 4 * b3
            one_w_sum = 1.0 / ( w1 + w2 + w3 )
            w1_normalized = w1 * one_w_sum
            w2_normalized = w2 * one_w_sum
            w3_normalized = w3 * one_w_sum
            return (w1_normalized * Variation1 + w2_normalized * Variation2 + w3_normalized * Variation3) / 6.0
        
        elif si_mininum < self.h_cutoff:
            a1 = (self.h_c + tau5 / ( s1 + epsilon_ ))**self.h_q
            a2 = (self.h_c + tau5 / ( s2 + epsilon_ ))**self.h_q
            a3 = (self.h_c + tau5 / ( s3 + epsilon_ ))**self.h_q
            one_a_sum = 1.0 / ( a1 + a2 + a3 )
            
            b1 = 0 if a1 * one_a_sum < CT_ else 1.0
            b2 = 0 if a2 * one_a_sum < CT_ else 1.0
            b3 = 0 if a3 * one_a_sum < CT_ else 1.0
        
            eta = self.h_eta
            w1 = (2 + eta) / 4 * b1
            w2 = (1 - eta) / 2 * b2
            w3 = eta / 4 * b3
            one_w_sum = 1.0 / ( w1 + w2 + w3 )
            w1_normalized = w1 * one_w_sum
            w2_normalized = w2 * one_w_sum
            w3_normalized = w3 * one_w_sum
            return (w1_normalized * Variation1 + w2_normalized * Variation2 + w3_normalized * Variation3) / 6.0
        else:          
            eta = self.l_eta
            w1 = (2 + eta) / 4
            w2 = (1 - eta) / 2
            w3 = eta / 4

            return (w1 * Variation1 + w2 * Variation2 + w3 * Variation3) / 6.0


class TENO5DUCROS_D:
    def __init__(
        self,
        coefficients: list
    ):
        self.eno = coefficients[0]
        self.d_linear = coefficients[1]
        self.v_linear = coefficients[2] 
        
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

        s11 = coef_smoothness_11_ * v2 + coef_smoothness_12_ * v3 + coef_smoothness_13_ * v4
        s12 = coef_smoothness_14_ * v2 + coef_smoothness_15_ * v4
        s1 = coef_smoothness_1_ * s11 * s11 + coef_smoothness_2_ * s12 * s12

        s21 = coef_smoothness_21_ * v3 + coef_smoothness_22_ * v4 + coef_smoothness_23_ * v5
        s22 = coef_smoothness_24_ * v3 + coef_smoothness_25_ * v4 + coef_smoothness_26_ * v5
        s2 = coef_smoothness_1_ * s21 * s21 + coef_smoothness_2_ * s22 * s22

        s31 = coef_smoothness_31_ * v1 + coef_smoothness_32_ * v2 + coef_smoothness_33_ * v3
        s32 = coef_smoothness_34_ * v1 + coef_smoothness_35_ * v2 + coef_smoothness_36_ * v3
        s3 = coef_smoothness_1_ * s31 * s31 + coef_smoothness_2_ * s32 * s32
        
        Variation1 = coef_stencils_1_ * v2 + coef_stencils_2_ * v3 + coef_stencils_3_ * v4
        Variation2 = coef_stencils_4_ * v3 + coef_stencils_5_ * v4 + coef_stencils_6_ * v5
        Variation3 = coef_stencils_7_ * v1 + coef_stencils_8_ * v2 + coef_stencils_9_ * v3
        
        epsilon_ = 1.0e-30
        tau5 = abs( s3 - s2 )
        CT_ = 1e-5
        
        a1 = (1 + tau5 / ( s1 + epsilon_ ))**6
        a2 = (1 + tau5 / ( s2 + epsilon_ ))**6
        a3 = (1 + tau5 / ( s3 + epsilon_ ))**6
        one_a_sum = 1.0 / ( a1 + a2 + a3 )
        
        b1 = 0 if a1 * one_a_sum < CT_ else 1.0
        b2 = 0 if a2 * one_a_sum < CT_ else 1.0
        b3 = 0 if a3 * one_a_sum < CT_ else 1.0

        if b1 < 0.1 or b2 < 0.1 or b3 < 0.1:            
            eta = self.eno
            w1 = (2 + eta) / 4 * b1
            w2 = (1 - eta) / 2 * b2
            w3 = eta / 4 * b3
            one_w_sum = 1.0 / ( w1 + w2 + w3 )
            w1_normalized = w1 * one_w_sum
            w2_normalized = w2 * one_w_sum
            w3_normalized = w3 * one_w_sum
            return (w1_normalized * Variation1 + w2_normalized * Variation2 + w3_normalized * Variation3) / 6.0
        else:
            eta = self.d_linear
            w1 = (2 + eta) / 4
            w2 = (1 - eta) / 2
            w3 = eta / 4
            return (w1 * Variation1 + w2 * Variation2 + w3 * Variation3) / 6.0

class TENO5DUCROS_V:
    def __init__(
        self,
        coefficients: list
    ):
        self.eno = coefficients[0]
        self.d_linear = coefficients[1]
        self.v_linear = coefficients[2]
        # self.ct = coefficients[3]
        
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

        s11 = coef_smoothness_11_ * v2 + coef_smoothness_12_ * v3 + coef_smoothness_13_ * v4
        s12 = coef_smoothness_14_ * v2 + coef_smoothness_15_ * v4
        s1 = coef_smoothness_1_ * s11 * s11 + coef_smoothness_2_ * s12 * s12

        s21 = coef_smoothness_21_ * v3 + coef_smoothness_22_ * v4 + coef_smoothness_23_ * v5
        s22 = coef_smoothness_24_ * v3 + coef_smoothness_25_ * v4 + coef_smoothness_26_ * v5
        s2 = coef_smoothness_1_ * s21 * s21 + coef_smoothness_2_ * s22 * s22

        s31 = coef_smoothness_31_ * v1 + coef_smoothness_32_ * v2 + coef_smoothness_33_ * v3
        s32 = coef_smoothness_34_ * v1 + coef_smoothness_35_ * v2 + coef_smoothness_36_ * v3
        s3 = coef_smoothness_1_ * s31 * s31 + coef_smoothness_2_ * s32 * s32
        
        Variation1 = coef_stencils_1_ * v2 + coef_stencils_2_ * v3 + coef_stencils_3_ * v4
        Variation2 = coef_stencils_4_ * v3 + coef_stencils_5_ * v4 + coef_stencils_6_ * v5
        Variation3 = coef_stencils_7_ * v1 + coef_stencils_8_ * v2 + coef_stencils_9_ * v3
        
        epsilon_ = 1.0e-30
        tau5 = abs( s3 - s2 )
        CT_ = 1e-10
        
        a1 = (1 + tau5 / ( s1 + epsilon_ ))**6
        a2 = (1 + tau5 / ( s2 + epsilon_ ))**6
        a3 = (1 + tau5 / ( s3 + epsilon_ ))**6
        one_a_sum = 1.0 / ( a1 + a2 + a3 )
        
        b1 = 0 if a1 * one_a_sum < CT_ else 1.0
        b2 = 0 if a2 * one_a_sum < CT_ else 1.0
        b3 = 0 if a3 * one_a_sum < CT_ else 1.0
  
        eta = self.v_linear
        w1 = (2 + eta) / 4 * b1
        w2 = (1 - eta) / 2 * b2
        w3 = eta / 4 * b3
        one_w_sum = 1.0 / ( w1 + w2 + w3 )
        w1_normalized = w1 * one_w_sum
        w2_normalized = w2 * one_w_sum
        w3_normalized = w3 * one_w_sum
        return (w1_normalized * Variation1 + w2_normalized * Variation2 + w3_normalized * Variation3) / 6.0