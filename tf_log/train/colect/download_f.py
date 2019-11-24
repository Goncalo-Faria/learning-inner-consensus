import wandb
from shutil import copyfile

api = wandb.Api()
run = api.run("graf/Gulbenkian/thxtgsrj")
start  = True
meta  = None
for file in run.files():
    fname = file.name
    fnames = fname.split(".")
    print(fname)
    if("checkpoint"==fname):
        file.download(replace=True)

    if fnames[0] == "model":
        if start :
            if fnames[-1] == "meta":
                meta = ".".join(fnames)
                start = False
        else:
            copyfile(meta, ".".join(fnames[:-1]) + ".meta")
            #print(meta)
            #print(".".join(fnames[:-1]) + ".meta")

        file.download(replace=True)
