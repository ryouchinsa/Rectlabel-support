import cv2
import numpy as np
from pycocotools import mask

def is_clockwise(contour):
    value = 0
    num = len(contour)
    for i, point in enumerate(contour):
        p1 = contour[i]
        if i < num - 1:
            p2 = contour[i + 1]
        else:
            p2 = contour[0]
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1]);
    return value < 0

def get_merge_point_idx(contour1, contour2):
    idx1 = 0
    idx2 = 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2);
            if distance_min < 0:
                distance_min = distance
                idx1 = i
                idx2 = j
            elif distance < distance_min:
                distance_min = distance
                idx1 = i
                idx2 = j
    return idx1, idx2

def merge_contours(contour1, contour2, idx1, idx2):
    contour = []
    for i in list(range(0, idx1 + 1)):
        contour.append(contour1[i])
    for i in list(range(idx2, len(contour2))):
        contour.append(contour2[i])
    for i in list(range(0, idx2 + 1)):
        contour.append(contour2[i])
    for i in list(range(idx1, len(contour1))):
        contour.append(contour1[i])
    contour = np.array(contour)
    return contour

def merge_with_parent(contour_parent, contour):
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)

def mask2polygon(image):
    contours, hierarchies = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contours_approx = []
    polygons = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        contours_approx.append(contour_approx)

    contours_parent = []
    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx < 0 and len(contour) >= 3:
            contours_parent.append(contour)
        else:
            contours_parent.append([])

    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx >= 0 and len(contour) >= 3:
            contour_parent = contours_parent[parent_idx]
            if len(contour_parent) == 0:
                continue
            contours_parent[parent_idx] = merge_with_parent(contour_parent, contour)

    contours_parent_tmp = []
    for contour in contours_parent:
        if len(contour) == 0:
            continue
        contours_parent_tmp.append(contour)

    polygons = []
    for contour in contours_parent_tmp:
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    return polygons 

if __name__ == '__main__':
    segmentation = {
        "counts": "[dVV67Pn2;F:F:G9K5J6K3L3N2M2O2M3O1N2N2O1N2N2N2O1N2N2N2N2O1N2N2N2N2N2O1N2N2N2N2N2O1N2N2N2N2N2O2M2N2N2O1N2O1N2O1N2O1N2O1N2O1N2O1N2O1N2O1N2O1N2N2O001N2O1N2O1O0O2O1N2O1O1N2O001N2O1N2O1O0O2O1N2O1O1N101O1N2O1N101O1N2O0O2O1O1N101O1N101N2O1O0O2O1N2O001N2O1O0O2O1N2O001N2O0O2O001N101O0O2O001N101O001N101O0O2O001N101O001N101O0O2O000O2O001nZMQHcc2o7W\\M[Hdc2g7T\\MeHgc2[7R\\MQIic2P7P\\MZIkc2g6n[MeImc2c8K4K3N1O1O1O2M2O1O1O2M2O1O2N1N2O1O2N0O1000000O101O00000O1000000O2O00000O1000000O2O0000000O1000000O2O00000O1000000O2O0000000O1O001N1O101N1O100O2N100O2N1O101O01O100O00100O001O1O1O0O2O1O001O1O1N101O1O1O001N2O0O2N2N2N1O2N2N2UOT]MRFmb2o9[]MfEhb2Z:f00O1000000O100001O0001O0001O000000000010O0000001O01O01O0000010O00001O0001O01O00010O01O0010O01O001O001O001O001O10O01O001O001O001O001O0010O01O001O001O1O001O001O002O0O1O2N1O1010O1O10O01O100O1O010O0001N10001N10001O0O=DO10O10O10O01000O0100O01000O010O01000O010O10O10O010O10O010O10O10O010O10O10O010O10O10O010O10O1O00000000000000001O0000000000000000000000001O0000000000000001O000000000000000O100000000O1000001O0O1000001O0O100000001N1000000O101O00000O1O1M4nMiZMiJYe2W5hZMfJZe2[5gZM`J\\e2`5eZM]J]e2c5eZMXJ^e2h5cZMUJ`e2j5bZMQJae2o5`ZMnIbe2S6_ZMiIce2W6^ZMeIee2[6]ZMaIee2_6V1O0001O000001O0010O0001O001j0UO3N1N2O2M2N10O0010O01O0O2M2O2N1N2J7G8I8G8SLbXMc0fg2[O[XM>lg2AUXM:Qh2DPXM9Sh2FoWM6Uh2HlWM4Xh2KiWM2Zh2NfWMO^h2OdWMM_h22bWMKbh23_WMKch24^WMJeh24]WMIeh26\\WMHgh26ZWMHhh27ZWMFih28XWMElh2:TWMDnh2;TWMBoh2<RWMBQi2<QWMARi2=oVMATi2=nVM@Ti2?mVM_OVi2?lVM^OWi2?kVM_OXi2?jVM]OZi2a0fVM^O]i2?eVM_O]i2`0dVM^O_i2`0aVM_Obi2>`VMAbi2=^VMCdi2;]VMDdi2:^VMEdi29\\VMGfi27[VMHgi25[VMJgi24[VMJfi25\\VMIfi24\\VMKhi21ZVMMoi2JSVM4Uj2CnUM;Sl2O2N2K5KRbTa5",
        "size": [3022, 4666]
    }
    m = mask.decode(segmentation) 
    m[m > 0] = 255
    polygons = mask2polygon(m)
    print(polygons)




