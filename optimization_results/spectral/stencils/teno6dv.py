from .base import epsilon_

coef_smoothness_1_ = 0.25
coef_smoothness_2_ = 13.0 / 12.0

coef_smoothness_11_ = 1.0
coef_smoothness_12_ = -4.0
coef_smoothness_13_ = 3.0
coef_smoothness_14_ = 1.0
coef_smoothness_15_ = -2.0
coef_smoothness_16_ = 1.0

coef_smoothness_21_ = 1.0
coef_smoothness_22_ = -1.0
coef_smoothness_23_ = 1.0
coef_smoothness_24_ = -2.0
coef_smoothness_25_ = 1.0

coef_smoothness_31_ = 3.0
coef_smoothness_32_ = -4.0
coef_smoothness_33_ = 1.0
coef_smoothness_34_ = 1.0
coef_smoothness_35_ = -2.0
coef_smoothness_36_ = 1.0

coef_stencils_1_ = 2
coef_stencils_2_ = -7
coef_stencils_3_ = 11
coef_stencils_4_ = -1
coef_stencils_5_ = 5
coef_stencils_6_ = 2
coef_stencils_7_ = 2
coef_stencils_8_ = 5
coef_stencils_9_ = -1
coef_stencils_10_ = 3
coef_stencils_11_ = 13
coef_stencils_12_  = -5
coef_stencils_13_  = 1   

multiplyer_stencils_1 = 1.0 / 6.0
multiplyer_stencils_2 = 1.0 / 12.0


class TENO6DUCROS_D:
    def __init__(
        self,
        eta_eno=0.8919,
        eta_v=0.4829,
        sensor_cutoff=0.1145,
        CT=1.0e-7,
        Cq=1,
        q=6
    ):
        self.eta_eno = eta_eno
        self.eta_v = eta_v
        self.sensor_cutoff = sensor_cutoff
        self.CT = CT
        self.Cq, self.q = Cq, q
        
    def apply(self, value):
        if not isinstance(value, list):
            raise Exception("Inputs must be a list.")
        elif len(value) != 6:
            raise Exception("Inputs must have at least 6 values.")
        else:
            v1 = value[0]
            v2 = value[1]
            v3 = value[2]
            v4 = value[3]
            v5 = value[4]
            v6 = value[5]
            
            Variation1 = coef_stencils_1_ * v1 + coef_stencils_2_ * v2 + coef_stencils_3_ * v3
            Variation2 = coef_stencils_4_ * v2 + coef_stencils_5_ * v3 + coef_stencils_6_ * v4
            Variation3 = coef_stencils_7_ * v3 + coef_stencils_8_ * v4 + coef_stencils_9_ * v5
            Variation4 = coef_stencils_10_ * v3 + coef_stencils_11_ * v4 + coef_stencils_12_ * v5 + coef_stencils_13_ * v6
            
            s11 = coef_smoothness_11_ * v1 + coef_smoothness_12_ * v2 + coef_smoothness_13_ * v3
            s12 = coef_smoothness_14_ * v1 + coef_smoothness_15_ * v2 + coef_smoothness_16_ * v3
            s1 = coef_smoothness_1_ * s11 * s11 + coef_smoothness_2_ * s12 * s12  # beta 0
            
            s21 = coef_smoothness_21_ * v2 + coef_smoothness_12_ * v4
            s22 = coef_smoothness_23_ * v2 + coef_smoothness_24_ * v3 + coef_smoothness_25_ * v4
            s2 = coef_smoothness_1_ * s21 * s21 + coef_smoothness_2_ * s22 * s22  # beta 1
            
            s31 = coef_smoothness_31_ * v3 + coef_smoothness_32_ * v4 + coef_smoothness_33_ * v5
            s32 = coef_smoothness_34_ * v3 + coef_smoothness_35_ * v4 + coef_smoothness_36_ * v5
            s3 = coef_smoothness_1_ * s31 * s31 + coef_smoothness_2_ * s32 * s32  # beta 2
            
            s4 = 1.0 / 240.0 * (
                v3   * (2107  * v3 - 9402  * v4 + 7042 * v5 - 1854 * v6) \
                + v4 * (11003 * v4 - 17246 * v5 + 4642 * v6) \
                + v5 * (7043  * v5 - 3882  * v6) \
                + 547 * v6 * v6
            ) # beta 3
            
            s7 = 1.0 / 120960 * (
                271779 * v1 * v1 + \
                v1 * (-2380800 * v2 + 4086352  * v3  - 3462252  * v4 + 1458762 * v5  - 245620  * v6) + \
                v2 * (5653317  * v2 - 20427884 * v3  + 17905032 * v4 - 7727988 * v5  + 1325006 * v6) + \
                v3 * (19510972 * v3 - 35817664 * v4  + 15929912 * v5 - 2792660 * v6) + \
                v4 * (17195652 * v4 - 15880404 * v5  + 2863984  * v6) + \
                v5 * (3824847  * v5 - 1429976  * v5) + \
                139633 * v6 * v6
            ) # beta 6
            
            tau6 = abs(s7 - 1/6 * (s1 + 4 * s2 + s3))
            
            a1 = (self.Cq + tau6 / (s1 + epsilon_))**self.q
            a2 = (self.Cq + tau6 / (s2 + epsilon_))**self.q
            a3 = (self.Cq + tau6 / (s3 + epsilon_))**self.q
            a4 = (self.Cq + tau6 / (s4 + epsilon_))**self.q
            one_a_sum = 1.0 / (a1 + a2 + a3 + a4)
            
            b1 = 0.0 if a1 * one_a_sum < self.CT else 1.0
            b2 = 0.0 if a2 * one_a_sum < self.CT else 1.0
            b3 = 0.0 if a3 * one_a_sum < self.CT else 1.0
            b4 = 0.0 if a4 * one_a_sum < self.CT else 1.0
            
            #dilatational
            eta = self.eta_eno
            w1 = 1/10 * eta * b1
            w2 = 3/10 * (eta+1) * b2
            w3 = 3/10 * b3
            w4 = 2/5 * (1- eta) * b4
            
            one_w_sum = 1.0 / ( w1 + w2 + w3 + w4)
            w1 = w1 * one_w_sum
            w2 = w2 * one_w_sum
            w3 = w3 * one_w_sum
            w4 = w4 * one_w_sum
            
            return (w1 * Variation1 + w2 * Variation2 + w3 * Variation3) * multiplyer_stencils_1 + w4 * Variation4 * multiplyer_stencils_2
            
            
class TENO6DUCROS_V:
    def __init__(
        self,
        eta_eno=0.8919,
        eta_v=0.4829,
        sensor_cutoff=0.1145,
        CT=1.0e-10,
        Cq=1,
        q=6
    ):
        self.eta_eno = eta_eno
        self.eta_v = eta_v
        self.sensor_cutoff = sensor_cutoff
        self.CT = CT
        self.Cq, self.q = Cq, q
        
    def apply(self, value):
        if not isinstance(value, list):
            raise Exception("Inputs must be a list.")
        elif len(value) != 6:
            raise Exception("Inputs must have at least 6 values.")
        else:
            v1 = value[0]
            v2 = value[1]
            v3 = value[2]
            v4 = value[3]
            v5 = value[4]
            v6 = value[5]
            
            Variation1 = coef_stencils_1_ * v1 + coef_stencils_2_ * v2 + coef_stencils_3_ * v3
            Variation2 = coef_stencils_4_ * v2 + coef_stencils_5_ * v3 + coef_stencils_6_ * v4
            Variation3 = coef_stencils_7_ * v3 + coef_stencils_8_ * v4 + coef_stencils_9_ * v5
            Variation4 = coef_stencils_10_ * v3 + coef_stencils_11_ * v4 + coef_stencils_12_ * v5 + coef_stencils_13_ * v6
            
            s11 = coef_smoothness_11_ * v1 + coef_smoothness_12_ * v2 + coef_smoothness_13_ * v3
            s12 = coef_smoothness_14_ * v1 + coef_smoothness_15_ * v2 + coef_smoothness_16_ * v3
            s1 = coef_smoothness_1_ * s11 * s11 + coef_smoothness_2_ * s12 * s12  # beta 0
            
            s21 = coef_smoothness_21_ * v2 + coef_smoothness_12_ * v4
            s22 = coef_smoothness_23_ * v2 + coef_smoothness_24_ * v3 + coef_smoothness_25_ * v4
            s2 = coef_smoothness_1_ * s21 * s21 + coef_smoothness_2_ * s22 * s22  # beta 1
            
            s31 = coef_smoothness_31_ * v3 + coef_smoothness_32_ * v4 + coef_smoothness_33_ * v5
            s32 = coef_smoothness_34_ * v3 + coef_smoothness_35_ * v4 + coef_smoothness_36_ * v5
            s3 = coef_smoothness_1_ * s31 * s31 + coef_smoothness_2_ * s32 * s32  # beta 2
            
            s4 = 1.0 / 240.0 * (
                v3   * (2107  * v3 - 9402  * v4 + 7042 * v5 - 1854 * v6) \
                + v4 * (11003 * v4 - 17246 * v5 + 4642 * v6) \
                + v5 * (7043  * v5 - 3882  * v6) \
                + 547 * v6 * v6
            ) # beta 3
            
            s7 = 1.0 / 120960 * (
                271779 * v1 * v1 + \
                v1 * (-2380800 * v2 + 4086352  * v3  - 3462252  * v4 + 1458762 * v5  - 245620  * v6) + \
                v2 * (5653317  * v2 - 20427884 * v3  + 17905032 * v4 - 7727988 * v5  + 1325006 * v6) + \
                v3 * (19510972 * v3 - 35817664 * v4  + 15929912 * v5 - 2792660 * v6) + \
                v4 * (17195652 * v4 - 15880404 * v5  + 2863984  * v6) + \
                v5 * (3824847  * v5 - 1429976  * v5) + \
                139633 * v6 * v6
            ) # beta 6
            
            tau6 = abs(s7 - 1/6 * (s1 + 4 * s2 + s3))
            
            a1 = (self.Cq + tau6 / (s1 + epsilon_))**self.q
            a2 = (self.Cq + tau6 / (s2 + epsilon_))**self.q
            a3 = (self.Cq + tau6 / (s3 + epsilon_))**self.q
            a4 = (self.Cq + tau6 / (s4 + epsilon_))**self.q
            one_a_sum = 1.0 / (a1 + a2 + a3 + a4)
            
            b1 = 0.0 if a1 * one_a_sum < self.CT else 1.0
            b2 = 0.0 if a2 * one_a_sum < self.CT else 1.0
            b3 = 0.0 if a3 * one_a_sum < self.CT else 1.0
            b4 = 0.0 if a4 * one_a_sum < self.CT else 1.0
            
            # vortical
            eta = self.eta_v
            w1 = 1/10 * eta * b1
            w2 = 3/10 * (eta+1) * b2
            w3 = 3/10 * b3
            w4 = 2/5 * (1- eta) * b4
            
            one_w_sum = 1.0 / ( w1 + w2 + w3 + w4)
            w1 = w1 * one_w_sum
            w2 = w2 * one_w_sum
            w3 = w3 * one_w_sum
            w4 = w4 * one_w_sum
            
            return (w1 * Variation1 + w2 * Variation2 + w3 * Variation3) * multiplyer_stencils_1 + w4 * Variation4 * multiplyer_stencils_2