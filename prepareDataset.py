#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:44:27 2017

@author: aradillas
"""
import sys
sys.path.append('RCNN/HTRTF/Modules')
from utils import initialize_log
import re
import os
import random
def load_char_list(char_list_path):
  charlist = []
  with open(char_list_path) as f:
    for l in f:
      charlist.append(l.strip())
  return charlist
  
def read_char_list(char_list_file_path, log_file_indicator):
    try:
        char_file = open(char_list_file_path)
    except IOError:
        print('No existe el fichero: ', char_list_file_path)
        log_file_indicator.write('\n Excepcion: No existe el fichero: ' + str(char_list_file_path) + '\n')
        log_file_indicator.close()
        sys.exit([1])
    else:
        char_list = []
        for l in char_file:
            char_list.append(l.strip())
        return char_list
def load_file_list_and_transcriptions_and_sizes_and_n_labels(file_list_path, char_list_path, base_path):
  charlist = load_char_list(char_list_path)
  file_list = []
  transcription_list = []
  transcriptionLenList = []
  targets = []
  size_list = []
  new_sequence = []
  trans = []
  with open(file_list_path) as f:
    for l in f:
      # IAM format
      if l.startswith("#"):
        continue
      sp = l.split()
      status = sp[1]
      assert status in ("ok", "err"), status
      assert len(sp) >= 9, l
      name = sp[0]
      text = "".join(sp[8:])
      # add space before '
      text = re.sub("([^|])'" , "\g<1>|'", text)
      width = int(sp[6])
      height = int(sp[7])
      if height < 1 or width < 1:
        continue
      size_list.append([int(l.split()[6]), int(l.split()[7])])
      s = name.split('-')
      trans.append(text)
      name = base_path + name + '.tif'
      text = text.replace('A', 'A ')
      text = text.replace('B', 'B ')
      text = text.replace('E', 'E ')
      text = text.replace('M', 'M ')
      text = text.replace('sp', 'sp ')
      trtext = text
      # text = text.replace('|', '| ')
      
    # ##convert to list split
      texttr = text.split()
      lentxt = len(texttr)
      
      tr = []
      cou = 0
      for idx in texttr:
        tr.append(charlist.index(idx))
        cou = cou + 1

      text = " ".join(str(x) for x in tr)
      # text = map(lambda c: charlist)index(c), text)
      # if pad_whitespace:
         # text = [charlist.index("sp")] + text + [charlist.index("sp")]
      file_list.append(name)
      transcription_list.append(text)
      # print(text)
      new_transcription = ''
      new_sequence = []
      for char in texttr:
          # print char
          if char in charlist:
               new_transcription += char
               new_sequence.append(charlist.index(char))
      targets.append(new_sequence)
      #print(new_sequence)
  
  # print 'size_list', size_list
  # print 'targets' , targets
  # print 'file_list : ', file_list
  # print 'transcription_list: sana ', transcription_list
  # print 'lentranss: ', transcriptionLenList 
  return file_list, trans, targets, size_list, len(charlist)

  
def extract_file_list(file_path, char_list_file_path, log_file_indicator, base_path):
    try:
        img_list_file = open(file_path)
    except IOError:
        print('No existe el fichero: ', file_path)
    else:
        char_list = load_char_list(char_list_file_path)  # ##read_char_list(char_list_file_path, log_file_indicator)
        log_file_indicator.write(str(len(char_list)) + ' chars considerated from this database: \n')
        log_file_indicator.write(str(char_list) + '\n')
        imgList = []
        transcription_list = []
        sequence_list = []
        size_list = []
        imgList, transcription_list, targets, size_list, charlist = load_file_list_and_transcriptions_and_sizes_and_n_labels(file_path, char_list_file_path, base_path)
        return imgList, transcription_list, targets, size_list, char_list

      
def extract_file_listold(file_path, char_list_file_path, log_file_indicator):
    try:
        img_list_file = open(file_path)
    except IOError:
        print('No existe el fichero: ', file_path)
    else:
        char_list = read_char_list(char_list_file_path, log_file_indicator)
        log_file_indicator.write(str(len(char_list)) + ' chars considerated from this database: \n')
        log_file_indicator.write(str(char_list) + '\n')
        imgList = []
        transcription_list = []
        sequence_list = []
        size_list = []
        n_ok = 0
        n_err_tr = 0
        n_err_simb = 0
        for l in img_list_file:
            if l.startswith('#'):
                continue
            if l.split()[1] in ['ok', 'err']:
                new_transcription = ''
                new_sequence = []
                for char in '|' + '|'.join(l.split()[8:]) + '|':
                    if char in char_list:
                        new_transcription += char
                        new_sequence.append(char_list.index(char))
                if new_transcription != '':
                    imgList.append(l.split()[0])
                    transcription_list.append(new_transcription)
                    sequence_list.append(new_sequence)
                    size_list.append([int(l.split()[6]), int(l.split()[7])])
                    n_ok += 1
                    log_file_indicator.write('Img: ' + str(l.split()[0]) + ' added. \tTranscription: \t' + str(' '.join(l.split()[8:])) + '\t-->\t' + str(new_transcription) + '\t\tSize: ' + str(size_list[-1]) + '\n')
                else:
                    log_file_indicator.write('Img: ' + str(l.split()[0]) + ' deleted. \tTranscription: \t' + str(' '.join(l.split()[8:])) + '\n')
                    n_err_simb += 1
            else:
                n_err_tr += 1 
        log_file_indicator.write('#' * 100 + '\n')
        log_file_indicator.write(str(n_ok) + ' secuencias a??adidas. \n' + str(n_err_tr) + ' secuencias eliminadas por posible segmentaci??n err??nea.\n' + str(n_err_simb) + ' secuencias eliminadas por ser palabras sin d??gitos ni letras (s??lo s??mbolos).\n')
        return imgList, transcription_list, sequence_list, size_list, char_list

def create_char_list_file(char_list_file_path, transcriptions_file_path, log_file_indicator):
    try:

        char_file = open(char_list_file_path, 'w')
    except IOError:
        print('No existe la ruta al fichero: ', char_list_file_path)
        log_file_indicator.write('\n Excepcion: No existe el fichero: ' + str(char_list_file_path) + '\n')
        log_file_indicator.close()
        sys.exit([1])
    else:
        try:
            transcriptions_file = open(transcriptions_file_path, 'r')
        except IOError:
            print('No existe el fichero: ', transcriptions_file)
            log_file_indicator.write('\n Excepcion: No existe el fichero: ' + str(transcriptions_file_path) + '\n')
            log_file_indicator.close()
            sys.exit([1])
        else:
            lines = transcriptions_file.readlines()
            char_list = []
            char_set = set()
            # Se extraen todos los caracteres de las lineas (separados por "-" y las palabras separadas por "|") y se anaden al conjunto aquellos que no estan todavia
            for line in lines:
                if line.startswith('#'):
                   continue
                if line.split()[1] == 'ok':
                    for word in ''.join(line.split()[8:]).split('|'):
                        for char in word:
                            char_list.append(char)
                            char_set = char_set | set(char_list)
            # Finalmente se anade el caracter espacio (|) que se elimino con el split()
            char_set = char_set | set('|')   
            # El conjunto resultante se pasa a lista y se almacena en un fichero llamado "charsWashington.txt" similara al chars.txt del IAM
            charlist = list(char_set)
            for char in charlist:
                char_file.write(char + '\n')
            char_file.close()   

        


def find_max_height(imgList, size_list):
    from math import ceil
    maxHeight = 0
    maxWidth = 0
    imgName = ''
    for ind, element in enumerate(size_list):
        if element[1] > maxHeight:
            maxHeight = element[1]
            imgName = imgList[ind]
    print(maxHeight, imgName)
    total64 = 0
    total128 = 0
    for ind, element in enumerate(size_list):
        if element[1] > 128:
            total64 += 1
        if element[1] > 180:
            total128 += 1
    print('Mayores que 180: ' + str(total128 / len(size_list)))
    print('Mayores que 128: ' + str(total64 / len(size_list)))
    for ind, element in enumerate(size_list):
        element[0] = ceil(element[0] * 128 / element[1])
        if element[0] > maxWidth:
            maxWidth = element[0]
            imgName = imgList[ind]
    print(maxWidth, imgName)
    total2048 = 0
    total1024 = 0
    for ind, element in enumerate(size_list):
        if element[0] > 4096:
            total1024 += 1
        if element[0] > 6000:
            total2048 += 1
    print('Mayores que 4096: ' + str(total1024 / len(size_list)))
    print('Mayores que 6000: ' + str(total2048 / len(size_list)))
    return maxHeight

def select_imgList_by_height(imgList, transcriptionList, sequenceList, sizeList, maxHeight, DataBasePath, log_file_indicator):
    from time import time
    from math import floor
    from sys import float_info
    import os.path

    
    eps = float_info.epsilon
    
    log_file_indicator.write('Analyzing database. Removing wrong data.\n')
    numImg_init = len(imgList)
    init_time = time()
    prev_percent = -1
    for ind, element in reversed(list(enumerate(sizeList))):
        time_elapsed = floor(1000 * (time() - init_time)) / 1000
        if floor(100 * (numImg_init - ind) / numImg_init) > prev_percent:
            prev_percent = floor(100 * (numImg_init - ind) / numImg_init)
            remaining_time = floor(1000 * (100 * time_elapsed / (prev_percent + eps) - time_elapsed)) / 1000
            print('Analyzing database. Removing wrong data. Time elapsed: ' + str(time_elapsed) + ' s. Remaining time: ' + str(remaining_time) + ' s.\n')
            # print('['+prev_percent*'|'+(100-prev_percent)*' '+'] '+str(prev_percent)+'%\n')
        if element[1] > maxHeight:
            log_file_indicator.write(str(imgList[ind]) + ' deleted because of having heigth ' + str(element[1]) + ' > ' + str(maxHeight) + ' px.\n')
            del imgList[ind]
            del transcriptionList[ind]
            del sequenceList[ind]
            del sizeList[ind]
        else: 
            img_name_split = imgList[ind].split('-')
            img_path = DataBasePath + img_name_split[0] + '/' + img_name_split[0] + '-' + img_name_split[1] + '/' + imgList[ind] + '.tif'
            
            if os.path.isfile(img_path):
                if [-1, -1] == list(sizeList[ind]):
                    log_file_indicator.write(str(imgList[ind]) + ' deleted because of the file ' + img_path + ' has size ' + str([1, 1]) + ' while the size indicated is ' + str(sizeList[ind]) + '.\n')
                    del imgList[ind]
                    del transcriptionList[ind]
                    del sequenceList[ind]
                    del sizeList[ind]
            else:
                log_file_indicator.write(str(imgList[ind]) + ' deleted because of not existing the file ' + img_path + '.\n')
                del imgList[ind]
                del transcriptionList[ind]
                del sequenceList[ind]
                del sizeList[ind]
     
    numImg_end = len(imgList)
    log_file_indicator.write(str(numImg_init - numImg_end) + ' deleted. ' + str(100 * (numImg_init - numImg_end) / numImg_init) + '% of the total.\n')
    log_file_indicator.write('Now the database contains ' + str(numImg_end) + ' words.\n')
    return imgList, transcriptionList, sequenceList, sizeList
 
def calculate_size_for_each_image(imgList, transcriptionList, sequenceList, sizeList, maxHeight, log_file_indicator):
    from math import ceil
    for ind, element in enumerate(sizeList):
        element[0] = ceil(element[0] * maxHeight / element[1])
        element[1] = maxHeight
    return imgList, transcriptionList, sequenceList, sizeList

def select_imgList_by_width(imgList, transcriptionList, sequenceList, sizeList, maxWidth, log_file_indicator):
    log_file_indicator.write('Applying select_imgList_by_width in order to remove big images.\n')
    numImg_init = len(imgList)
    for ind, element in reversed(list(enumerate(sizeList))):
        if element[0] > maxWidth:
            log_file_indicator.write(str(imgList[ind]) + ' deleted because of having width ' + str(element[0]) + ' > ' + str(maxWidth) + ' px.\n')
            del imgList[ind]
            del transcriptionList[ind]
            del sequenceList[ind]
            del sizeList[ind]
    numImg_end = len(imgList)
    # log_file_indicator.write(str(numImg_init - numImg_end) + ' deleted. ' + str(100 * (numImg_init - numImg_end) / numImg_init) + '% of the total.\n')
    # log_file_indicator.write('Now the database contains ' + str(numImg_end) + ' words.\n')
    return imgList, transcriptionList, sequenceList, sizeList

def create_selection_list(selection_file, imgList, transcriptionList, sequenceList, sizeList, setName, log_file_indicator):    

    root_selection_list = load_char_list(selection_file)
    # print(root_selection_list)
    selected_imgList = []
    selected_transcriptionList = []
    selected_sequenceList = []
    selected_sizeList = []
    selected_transcriptionLenList = []
    total = 0
    for imgName, transcription, sequence, size in zip(imgList, transcriptionList, sequenceList, sizeList):
        print imgName
        head, tail = os.path.split(imgName)
        tail = tail.replace('.tif', '')
        imgName = tail
        # print(imgName)
        if root_selection_list.count(imgName) > 0  :
            print(imgName)
            selected_imgList.append(imgName)
            selected_transcriptionList.append(transcription)
            selected_sequenceList.append(sequence)
            selected_sizeList.append(size)
            selected_transcriptionLenList.append(len(sequence))
            log_file_indicator.write(str(imgName) + ' added to set ' + setName + '\n')
            total += 1
            print(total)
    log_file_indicator.write('\n' + str(total) + ' sequences added to set ' + setName + '\n')  
    
    return selected_imgList, selected_transcriptionList, selected_sequenceList, selected_sizeList, selected_transcriptionLenList, setName

def hdf5_strings(handle, name, data):
  import h5py
  try:
    S = max([len(d) for d in data])
    dset = handle.create_dataset(name, (len(data),), dtype="S" + str(S))
    dset[...] = data
  except Exception:
    dt = h5py.special_dtype(vlen=unicode)
    del handle[name]
    dset = handle.create_dataset(name, (len(data),), dtype=dt)
    dset[...] = data

def write_h5(h5_out_path, selected_imgList, selected_transcriptionList, selected_sequenceList, selected_sizeList, selected_transcriptionLenList, charlist, maxHeight, maxWidth, rutaDataBase, log_file_indicator):
    import numpy as np
    from PIL import Image
    from math import ceil
    import h5py
    
    with h5py.File(h5_out_path, "w") as f:
        f.attrs["numSeqs"] = len(selected_imgList)
        type_string = h5py.special_dtype(vlen=str) 
        
        labels_ds = f.create_dataset('labels', (len(charlist),), dtype=type_string)
        labels_ds[...] = charlist
        
        imgName_ds = f.create_dataset('data/imgNames', (len(selected_imgList),), dtype=type_string)
        imgName_ds[...] = selected_imgList
        
        transcription_ds = f.create_dataset('data/transcriptions', (len(selected_transcriptionList),), dtype=type_string)
        transcription_ds[...] = selected_transcriptionList
        
        targetsLengths = []
        for seq in selected_sequenceList:
            targetsLengths.append(len(seq))
            
        targetsLength_ds = f.create_dataset('data/targetsLengths', (len(targetsLengths),))
        targetsLength_ds[...] = targetsLengths
        
        sequenceList_np = np.concatenate(selected_sequenceList)
        
        targets_ds = f.create_dataset('data/targets', (len(sequenceList_np),))
        targets_ds[...] = sequenceList_np
        
        f['data/sizes'] = selected_sizeList
        f['data/trans_len'] = selected_transcriptionLenList
        # for seq in selected_sequenceList:
            # targetsLengths.append(len(seq))
            
        # targetsLength_ds = f.create_dataset('data/targetsLengths', (len(targetsLengths),))
        # targetsLength_ds[...] = targetsLengths
        
        
        # sequenceList_np = np.concatenate(selected_sequenceList)
        
        # targets_ds = f.create_dataset('data/targets', (len(sequenceList_np),))
        # targets_ds[...] = sequenceList_np
        
        # f['data/sizes'] = selected_sizeList
        # f['data/trans_len'] = targetsLengths
    log_file_indicator.write('\nFile ' + str(h5_out_path) + ' created. With datasets: \n')
    
    def printlog(name):
        log_file_indicator.write(str(name) + '\n')

    with h5py.File(h5_out_path, "r") as f:
        f.visit(printlog)
    
def create_dataset_h5_and_csv_file(selection_file, imgList, transcriptionList, sequenceList, sizeList, charlist, maxHeight, maxWidth, rutaDataBase, log_file_indicator):
    import pandas as pd
    from math import floor
    from time import time
    from sys import float_info
    
    eps = float_info.epsilon
    setName = str(selection_file.split('/')[-1].split('.')[0])
    selected_imgList, selected_transcriptionList, selected_sequenceList, selected_sizeList, selected_transcriptionLenList, setName = create_selection_list(selection_file, imgList, transcriptionList, sequenceList, sizeList, setName, log_file_indicator)
    csv_out_path = '/'.join(selection_file.split('/')[:-1]) + '/' + setName + '.csv'
    h5_out_path = '/'.join(selection_file.split('/')[:-1]) + '/' + setName + '.h5'
    df = pd.DataFrame(columns=['imgName', 'transcription', 'sequence', 'size', 'trans_length'])
    set_size = len(selected_imgList)
    prev_percent = -1
    init_time = time()
    for ind, (imgName, transcription, sequence, size, trans_len) in enumerate(zip(selected_imgList, selected_transcriptionList, selected_sequenceList, selected_sizeList, selected_transcriptionLenList)):
        df.loc[ind] = [imgName, transcription, sequence, size, trans_len]
        time_elapsed = floor(1000 * (time() - init_time)) / 1000
        if floor(100 * ind / set_size) > prev_percent:
            prev_percent = floor(100 * ind / set_size)
            remaining_time = floor(1000 * (100 * time_elapsed / (prev_percent + eps) - time_elapsed)) / 1000
            print('Creating ' + setName + ' dataset. Time elapsed: ' + str(time_elapsed) + ' s. Remaining time: ' + str(remaining_time) + ' s.\n')
            # print('['+prev_percent*'|'+(100-prev_percent)*' '+'] '+str(prev_percent)+'%\n')
    log_file_indicator.write('Saved ' + setName + ' information in ' + csv_out_path + ' dataset. \n')
    df.to_csv(csv_out_path)
    write_h5(h5_out_path, selected_imgList, selected_transcriptionList, selected_sequenceList, selected_sizeList, selected_transcriptionLenList, charlist, maxHeight, maxWidth, rutaDataBase, log_file_indicator)
 

					
			 
 
		
 

def main():
 
	# Set the path to image folder
	filepath = 'HTRTF/Projects/MADCAT-WriterDep-Global/allwriters.txt'
	alwriters=''
	setnumber='Set'
	writercode=''
	DataBasePath = 'RCNN/imagesMADCAT/'
 	###concat train and valid files to two files train and valid
	import glob
	import os
	# for f in glob.glob("HTRTF/Projects/MADCAT-WriterDep-Global/Set/*/list_train"):
		# os.system("cat "+f+" >> HTRTF/Projects/MADCAT-WriterDep-Global/Set/list_train")
	# for f in glob.glob("HTRTF/Projects/MADCAT-WriterDep-Global/Set/*/list_valid"):
		# os.system("cat "+f+" >> HTRTF/Projects/MADCAT-WriterDep-Global/Set/list_valid")

	###end concat
	with open(filepath) as fp:
	    
	   cnt = 1
	   for line in fp:
			print("Line {}: {}".format(cnt, line.strip()))
			 
			writercode=line	   
			writercode=writercode.replace('\r', '')
			writercode=writercode.replace('\n', '')
			img_list_file = 'HTRTF/Projects/MADCAT-WriterDep-Global/Set/lines.txt'
			
			# Set paths to the .txt files containing the id of the images on each dataset

			validation2_list_file = 'HTRTF/Projects/MADCAT-WriterDep-Global/Set/list_valid'
			train_list_file = 'Projects/MADCAT-WriterDep-Global/Set/list_train'
			test_list_file = 'HTRTF/Projects/MADCAT-WriterDep-Global/'+setnumber+'/'+writercode+'/list_test'
			 
			
			
			# Auxiliar files 
			char_list_file_path = 'HTRTF/Projects/MADCAT-WriterDep-Global/'+setnumber+'/CHAR_LIST'
			manage_database_log_path = 'HTRTF/Projects/MADCAT-WriterDep-Global/'+setnumber+'/'+writercode+'/read_database_log.txt'
			maxHeight = 70
			maxWidth = 1000
				 
			log_file_indicator = initialize_log(manage_database_log_path)

			imgList, transcriptionList, sequenceList, sizeList, char_list = extract_file_list(img_list_file, char_list_file_path, log_file_indicator, DataBasePath)
		  
			calculate_size_for_each_image(imgList, transcriptionList, sequenceList, sizeList, maxHeight, log_file_indicator)
			#imgList, transcriptionList, sequenceList, sizeList = select_imgList_by_width(imgList, transcriptionList, sequenceList, sizeList, maxWidth, log_file_indicator)
			
			# Creating datasets
		 
			create_dataset_h5_and_csv_file(validation2_list_file, imgList, transcriptionList, sequenceList, sizeList, char_list, maxHeight, maxWidth, DataBasePath, log_file_indicator)
			create_dataset_h5_and_csv_file(train_list_file, imgList, transcriptionList, sequenceList, sizeList, char_list, maxHeight, maxWidth, DataBasePath, log_file_indicator)
			#create_dataset_h5_and_csv_file(test_list_file, imgList, transcriptionList, sequenceList, sizeList, char_list, maxHeight, maxWidth, DataBasePath, log_file_indicator)
		 

			log_file_indicator.close()
    
if __name__ == "__main__":
    main()
