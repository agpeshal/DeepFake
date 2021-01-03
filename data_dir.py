
def data_dir_list():
    splits = ["train", "test", "val"]

    for split in splits:
        f = open(f"images/{split}.txt", 'r')
        out = open(f"images/{split}_dir.txt", 'w')
        for line in f.readlines():
            if "/0.png" in line:
                path = "/".join(line.split('/')[:3])
                label = line[-2]
                out.write("%s %s\n" % (path, label))
        
        f.close()
        out.close()

if __name__ == "__main__":
    data_dir_list()
