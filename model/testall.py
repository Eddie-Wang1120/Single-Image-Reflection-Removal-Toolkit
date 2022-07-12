import os
import warnings

import cv2
import matplotlib.pyplot as plt

def BDN():
    os.system('rm ./model/result.txt')
    warnings.warn("deprecated", DeprecationWarning)
    os.chdir(os.getcwd() + "/model/output/")
    os.system("rm -rf *")
    os.chdir("..")
    print("BDN testing")
    os.chdir(os.getcwd() + "/BDN/")
    fc = open(os.path.join('../', 'result.txt'), 'a')
    fc.write('BDN')
    fc.write('\n')
    fc.close()
    os.system('bash ./test.sh')
    os.chdir("..")

def IBCLN():
    print("IBCLN testing")
    os.chdir(os.getcwd() + "/IBCLN/")
    fc = open(os.path.join('../', 'result.txt'), 'a')
    fc.write('IBCLN')
    fc.write('\n')
    fc.close()
    os.system('bash ./test.sh')
    os.chdir("..")

def IBLNN():
    print("IBLNN testing")
    os.chdir(os.getcwd() + "/IBLNN/")
    fc = open(os.path.join('../', 'result.txt'), 'a')
    fc.write('IBLNN')
    fc.write('\n')
    fc.close()
    os.system('bash ./test.sh')
    os.chdir("..")

def PRN():
    print("PRN testing")
    os.chdir(os.getcwd() + "/PRN/")
    fc = open(os.path.join('../', 'result.txt'), 'a')
    fc.write('PRN')
    fc.write('\n')
    fc.close()
    os.system('bash ./test.sh')
    os.chdir("..")

def RR():
    print("RR testing")
    os.chdir(os.getcwd() + "/RR/")
    fc = open(os.path.join('../', 'result.txt'), 'a')
    fc.write('RR')
    fc.write('\n')
    fc.close()
    os.system('bash ./test.sh')
    os.chdir("..")
    os.chdir('..')

def resultResolute():
    os.chdir(os.getcwd() + "/model/")
    fc = open('result.txt')
    lines = fc.readlines()

    i = 0
    strlist = []
    psnrlist = []
    ssimlist = []

    for line in lines:
        if i % 2 == 0:
            strlist.append(line)
        else:
            psnrlist.append(float(line[11:20]))
            ssimlist.append(float(line[27:35]))
        i = i + 1

    plt.bar(range(len(psnrlist)), psnrlist, color='r', tick_label=strlist, label='psnr')
    plt.title('psnr')
    # plt.show()
    plt.bar(range(len(ssimlist)), ssimlist, color='b', tick_label=strlist, label='ssim')
    plt.title('ssim')
    # plt.show()
    os.chdir("..")


# os.system('rm ./result.txt')
# warnings.warn("deprecated", DeprecationWarning)
# os.chdir(os.getcwd() + "/output/")
# os.system("rm -rf *")
# os.chdir("..")
# print("BDN testing")
# os.chdir(os.getcwd() + "/BDN/")
# fc = open(os.path.join('../', 'result.txt'), 'a')
# fc.write('BDN')
# fc.write('\n')
# fc.close()
# os.system('bash ./test.sh')
# os.chdir("..")

    # print("IBCLN testing")
    # os.chdir(os.getcwd() + "/IBCLN/")
    # fc = open(os.path.join('../', 'result.txt'), 'a')
    # fc.write('IBCLN')
    # fc.write('\n')
    # fc.close()
    # os.system('bash ./test.sh')
    # os.chdir("..")
    #
    # print("PRN testing")
    # os.chdir(os.getcwd() + "/PRN/")
    # fc = open(os.path.join('../', 'result.txt'), 'a')
    # fc.write('PRN')
    # fc.write('\n')
    # fc.close()
    # os.system('bash ./test.sh')
    # os.chdir("..")
    #
    # print("RR testing")
    # os.chdir(os.getcwd() + "/RR/")
    # fc = open(os.path.join('../', 'result.txt'), 'a')
    # fc.write('RR')
    # fc.write('\n')
    # fc.close()
    # os.system('bash ./test.sh')
    # os.chdir("..")
    #
    # fc = open('result.txt')
    # lines = fc.readlines()
    #
    # i = 0
    # strlist = []
    # psnrlist = []
    # ssimlist = []
    #
    # for line in lines:
    #     if i % 2 == 0:
    #         strlist.append(line)
    #     else:
    #         psnrlist.append(float(line[11:20]))
    #         ssimlist.append(float(line[27:35]))
    #     i = i + 1
    #
    # plt.bar(range(len(psnrlist)), psnrlist, color='r', tick_label=strlist, label='psnr')
    # plt.title('psnr')
    # plt.show()
    # plt.bar(range(len(ssimlist)), ssimlist, color='b', tick_label=strlist, label='ssim')
    # plt.title('ssim')
    # plt.show()
    # os.chdir("..")
    # def showimg(name, fimg, a, b, c):
    #     img = cv2.imread(fimg)
    #     plt.subplot(a, b, c)
    #     plt.title(name)
    #     plt.xticks([])  # remove ticks
    #     plt.yticks([])
    #     # The color channel order in matplotlib is [R, G, B]
    #     # The color channel order in opencv is [B, G, R]
    #     plt.imshow(img[:, :, ::-1])
    #
    # # showimg('BDN_input', './dataset/I/1.jpg', 4,3,1)
    # # showimg('BDN_trans', './output/BDN/T_1.jpg', 4,3,2)
    # # showimg('BDN_output', './output/BDN/B_1.jpg', 4,3,3)
    # #
    # # showimg('IBCLN_input', './dataset/I/1.jpg', 4,3,4)
    # # showimg('IBCLN_trans', './output/IBCLN/test_final/images/1_real_T_00.jpg', 4,3,5)
    # # showimg('IBCLN_output', './output/IBCLN/test_final/images/1_fake_Ts_03.jpg', 4,3,6)
    # #
    # # showimg('PRN_input', './dataset/I/1.jpg', 4,3,7)
    # # showimg('PRN_trans', './output/BDN/T_1.jpg', 4,3,8)
    # # showimg('PRN_output', './output/PRN/1_T.jpg', 4,3,9)
    # #
    # # showimg('RR_input', './dataset/I/1.jpg', 4,3,10)
    # # showimg('RR_trans', './output/BDN/T_1.jpg', 4,3,11)
    # # showimg('RR_output', './output/RR/dataset/1_pred_T.jpg', 4,3,12)
    # #
    # # plt.show()

