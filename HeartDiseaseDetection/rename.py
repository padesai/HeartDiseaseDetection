import glob
import shutil
import os

for i in range(501,701):
    for filename in glob.glob("C:/Users/pgdes/NewDirectory/validate/"+ str(i) +"/study/4ch_*/*.dcm"):
        basefile = os.path.basename(filename)
        splitbasefile = basefile.split("-")
        file = splitbasefile[2]
        filenum = file.split(".")
        filenum = str(int(filenum[0]))
        shutil.copyfile(filename, "training_data/IM_" + str(i) + "_" + filenum +"_.dcm")

