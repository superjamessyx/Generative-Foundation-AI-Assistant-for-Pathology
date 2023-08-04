import json
import os
import platform
import subprocess

plat = platform.system().lower()
import shutil

from bs4 import BeautifulSoup
from tqdm import tqdm
import re


def convert_space_path(t):
    t = "/ ".join(t.split(' '))
    return t


def check_keywords(caption, key_words=['Figure', 'Fig', 'Image', 'Figures']):
    for ki in key_words:
        if ki.lower() in caption.lower():
            return True

    return False


#
# def check_keywords(caption, key_words=['FIGURE', 'Figure', 'Fig', 'Image','Img']):
#     return re.sub('python', 'PHP', caption, flags=re.IGNORECASE)

def parse_html(file_path, key_words=['FIGURE', 'Figure', 'Fig', 'Image', 'FIG', 'FIGURES', ' Fig', 'FiGure', 'figUre']):
    with open(file_path, 'r', encoding='utf-8') as f:
        bs = BeautifulSoup(f, "html.parser")
    all_p = bs.select('p')
    # ignore short text of p
    all_p = [i for i in all_p if i.getText().__len__() > 15 or check_keywords(i.getText()) or i.select('img')]
    # pi_idx=0
    # for pi in range(len(all_p)):
    json_out = {'anotations': []}
    html_dir = os.path.dirname(file_path)
    image_path = os.path.join(html_dir, 'media')
    image_path_covert = os.path.join(html_dir, 'media_convert')
    os.makedirs(image_path_covert, exist_ok=True)

    # for (pi_idx, pi) in enumerate(all_p):
    for (pi_idx, pi) in enumerate(tqdm(all_p)):
        # if pi.is_empty_element is not True:
        # current = all_p[pi]
        # pi = all_p[pi_idx]
        if pi.select('img'):
            # for i in pi.select('img'):
            #     img_list.
            step = 0
            img_list = pi.select('img')
            # if len(img_list)>1:
            #     while step <= 5 and  and pi_idx + step < all_p.__len__() :
            #         next_p =  all_p[pi_idx + step]
            #         label_list = next_p.select('strong')
            pair_list =[]
            caption = 'Valid caption'
            base_caption = caption
            for (img_idx, img_i) in enumerate(img_list):
                image_file_name = img_i.attrs['src'].split('/')[-1]
                image_id = image_file_name.split('.')[0]
                # if image_id == "image11":
                #     a=1
                image_format = image_file_name.split('.')[1]

                if img_i.has_attr('style'):
                    img_size = img_i.attrs['style']
                    img_size = [float(s) for s in re.findall(r'-?\d+\.?\d*', img_size)]

                    if img_size.__len__() < 2 or img_size[0] / img_size[1] > 2.9:
                        continue

                caption = 'Valid caption'
                pair = {}
                reach_next_img = False
                # while pi_idx + step < all_p.__len__():
                while step <= 5 * len(pi.select('img')) and pi_idx + step < all_p.__len__():
                    next_p = all_p[pi_idx + step]
                    if step>0 and next_p.select('img'):
                        reach_next_img = True
                        break
                    # if next_p.select('span') or next_p.select('strong') or next_p.select('p'):
                    caption = next_p.getText()

                    caption = caption.replace('\n', ' ')
                    caption = caption.replace('\u25a0', '')  # replace "■"n
                    caption = caption.expandtabs()
                    caption = re.sub(' +', ' ', caption)
                    # caption = next_p.getText(strip=True)
                    if check_keywords(caption):

                        # replace \n with space
                        if next_p.select('strong'):
                            key = next_p.select('strong')[0].getText()
                        else:
                            key = ' '.join(caption.split(' ')[:2])
                        key = key.replace('\n', ' ')
                        key = key.expandtabs()
                        key = re.sub(' +', ' ', key)
                        # if key=='FIG. 3.':
                        #     a=1
                        if len(key.split(' ')) > 2:
                            key = ' '.join(key.split(' ')[:2])
                        if check_keywords(key):
                            base_caption = caption
                            base_key = key
                            # os.rename(os.path.join(image_path,image_file_name), os.path.join(image_path_covert, key+'.'+image_format))\
                            ori_file = os.path.join(image_path, image_file_name)
                            convert_file = os.path.join(image_path_covert, key + '.' + image_format)
                            if not os.path.exists(convert_file):
                                shutil.copyfile(ori_file, convert_file)
                            else:
                                convert_file = os.path.join(image_path_covert,
                                                            key + "-%s" % (img_idx+1) + '.' + image_format)
                                shutil.copyfile(ori_file, convert_file)
                            # if plat == 'windows':
                            #     os.system('copy %s %s' % (os.path.join(image_path,image_file_name), os.path.join(image_path_covert, key+'.'+image_format)))
                            # elif plat == 'linux':
                            #     os.system('cp %s %s' % (os.path.join(image_path,image_file_name), os.path.join(image_path_covert, key+'.'+image_format)))
                        else:
                            # raise ValueError
                            step += 1
                            caption = 'Valid caption'
                            continue
                        # caption = caption.replace('\n',' ')
                        # # caption = caption.replace('      ', ' ')
                        # # caption = caption.replace('“', '"')
                        # caption = caption.expandtabs()
                        # caption = re.sub(' +', ' ', caption)
                        step += 1
                        break
                    else:
                        step += 1
                        caption = 'Valid caption'
                # else:
                #     step += 1
                if base_caption == 'Valid caption':
                    continue
                if reach_next_img and len(img_list)<2:
                    continue
                # elif len(img_list)>1:

                # pair['image_id'] = key
                if check_keywords(caption):
                    pair['image_id'] = key
                    pair['caption'] = caption
                else:
                    pair['image_id'] = base_key
                    pair['caption'] = base_caption
                    ori_file = os.path.join(image_path, image_file_name)
                    convert_file = os.path.join(image_path_covert,
                                                base_key + "-%s" % (img_idx + 1) + '.' + image_format)
                    shutil.copyfile(ori_file, convert_file)
                # pair_list.append(pair)
                json_out['anotations'].append(pair)
    return json_out


def batch_parse_html(cdir=os.getcwd(), key_words=['Fig', 'Figure']):
    # html_files = []
    for root, _, file in os.walk(cdir):

        for fi in file:
            if fi.endswith('.html'):
                # html_files.append()
                print("Processing HTML: %s" % fi)
                json_file_name = os.path.join(root, fi[:-5] + ".json")
                if os.path.exists(json_file_name):
                    print("Exists......")
                    continue
                html_path = os.path.join(root, fi)
                json_out = parse_html(html_path)
                with open(json_file_name, 'w', encoding='utf-8') as jf:
                    jf.write(json.dumps(json_out, ensure_ascii=False,indent=3))
                # with open(json_file_name, 'w',encoding='utf-8') as jf:
                #     jf.write(json.dumps(json_out))


def convert_html_from_doc(cdir=os.getcwd()):
    # cdir = os.getcwd()
    print("current dir: %s" % cdir)

    doc_list = [i for i in os.listdir(cdir) if i.endswith('.docx')]

    for di in doc_list:
        book_name = di.split('.')[0]

        print("Processing book DOCX file: %s" % book_name)
        if os.path.exists(os.path.join(book_name, book_name + '.html')):
            print("Exists......")
            continue
        os.makedirs(book_name, exist_ok=True)
        # target_path = convert_space_path(os.path.join(cdir, book_name))
        target_path = os.path.join(cdir, book_name)
        # target_path_1 = "\"" + target_path + "\""
        doc_path = "\"" + os.path.join(cdir, di) + "\""
        # html_path = "\"" + os.path.join(target_path, book_name + '.html') + "\""
        # command = r"pandoc --extract-media %s -s %s -o %s" % (
        #     "\".\\" + os.path.join(book_name) + "\"",
        #     # "\"" + book_name + "\"",
        #     "\".\\" + di + "\"",
        #     "\".\\" + os.path.join(book_name, book_name + '.html') + "\""
        # )
        os.chdir(book_name)
        command = "pandoc --extract-media \"./\" --standalone %s --output %s" % (
            doc_path,
            "\".\\" + book_name + '.html' + "\"",
        )
        command_process_obj = subprocess.run(command, stdout=subprocess.PIPE)
        os.chdir("..\\")
        # print("Move doc to %s" % target_path)
        # shutil.move(os.path.join(cdir, di), target_path)
        # print("Move doc to %s" % target_path)
        # shutil.move(os.path.join(cdir, di), target_path)


# def generate_annotation()
if __name__ == '__main__':
    key_words = ['FIGURE', 'Figure', 'Fig', 'Image', 'FIG', 'FIGURES', ' Fig', 'FiGure', 'figUre']
    convert_html_from_doc()
    print("--------------------------Batch prasing HTML----------------------")
    batch_parse_html(key_words=key_words)
