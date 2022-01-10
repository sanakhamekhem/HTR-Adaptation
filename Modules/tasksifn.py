
import numpy as np
from utils import shuffle_set, get_batch, print_valid_results, seconds_to_days_hours_min_sec, initialize_from_AACHEN, pack_images, levenshtein
from DataAugmentation import distort
from math import ceil, floor
import tensorflow as tf
import time
import codecs
from sys import float_info
eps=float_info.epsilon
import os
from io import StringIO

import codecs

def loadRNNOutput(logs):
    "load RNN output from csv file. Last entry in row terminated by semicolon."
    data = StringIO(logs.decode('utf-8'))
    return np.genfromtxt(data, delimiter=';')[:, :-1]

def createArk(logs, labs):
	data = loadRNNOutput(logs)
	mat = data  # #softmax(data)
 
	r = ''
	precision = 30

	for row in mat:
		# max value
		# if you want to iterate over the elements in the row do:
		x = ''
		i = 0
		for element in row:
			ww = str(np.round(element, precision))
			if i == 1:
				x = x + ' ' + '0.0' + ww
			else:
				x = x + ' ' + ww
		
		ww = x.strip()
   
		L = ww.split()
		
		ind_space = labs.index('<SPACE>')
		ind_sp = labs.index('sp')
		ind_triet = labs.index('<ep>')

		#L[ind_space + 1] = L[ind_sp + 1]
		del L[ind_triet+1]
		sq = ' '.join(L)
		

		r = r + sq + '\n'
		
	ark = r + ']'
	ark = ark.replace('\n]', ' ]')
	return ark
	

def train(step, network, num_epochs, batchSize, learning_rate, trainSet, imageHeight, imageWidth, num_classes, log_indicator, models_path, train_writer, transferFLAG = False, dataAugmentation = False):        
    
    train_nameList, train_inputs, train_targetList, train_seqLengths, train_heights, train_transcriptionList, train_transcriptionsLenList = trainSet
    
    trainSetSize=len(train_nameList)
    n_train_batches=ceil(trainSetSize/batchSize)
       
    graph, saver, inputs_mask_ph, seq_len_ph, targets_ph, targets_len_ph, learning_rate_ph,  n_batches_ph, setTotalChars_ph, previousEDabs_ph, previousEDnorm_ph, previousCost_ph, optimizer, batch_cost, cost, errors, ED, predictions, merged = network.create(imageHeight,imageWidth, num_classes, False)
    
    if type(inputs_mask_ph) == list:
        mask_ph=inputs_mask_ph[1]
        inputs_ph=inputs_mask_ph[0]
    else:
        inputs_ph=inputs_mask_ph

    if type(saver) == list:
        saver=saver[0]

        
    with tf.Session(graph=graph) as sess:
             
        if step==0 and not transferFLAG:
            train_writer.add_graph(sess.graph) 
            text=('\nInitializing weights randomly\n')
            print(text)
            log_indicator.write(text)
            tf.global_variables_initializer().run()
        else:
            saver.restore(sess=sess,save_path=tf.train.latest_checkpoint(models_path))

        for epoch in range(num_epochs):
            totalCost=0
            epoch_start=time.time()
            prev_percent=-1

            if dataAugmentation:
                      train_inputs_copy = distort(train_inputs)
                      train_inputs_copy = pack_images(train_inputs_copy, imageHeight, imageWidth)
            else:
                      train_inputs_copy = list(train_inputs)

            for batch in range(int(n_train_batches)):
                

                train_nameList_shuffled, train_inputs_shuffled, train_targetList_shuffled, train_seqLengths_shuffled, train_heihts_shuffled, train_transcriptionList_shuffled, train_transcriptionsLenList_shuffled=shuffle_set(train_nameList, train_inputs_copy, train_targetList, train_seqLengths, train_heights, train_transcriptionList, train_transcriptionsLenList)
                
                
                trainBatchNameList, trainBatchInputs, trainBatchTargetSparse, trainBatchSeqLengths, trainHeights, trainBatchTranscriptions, trainBatchTransLen=get_batch(batchSize, train_nameList_shuffled, train_inputs_shuffled, train_targetList_shuffled, train_seqLengths_shuffled, train_heihts_shuffled, train_transcriptionList_shuffled, train_transcriptionsLenList_shuffled)
                
                
        
                feed = {inputs_ph: trainBatchInputs,
                        targets_ph: trainBatchTargetSparse,
                        targets_len_ph: trainBatchTransLen,
                        seq_len_ph: trainBatchSeqLengths,
                        learning_rate_ph: learning_rate,
                        n_batches_ph: n_train_batches,
                        previousCost_ph: totalCost}

                if type(inputs_mask_ph) == list:
                   mask=np.zeros([len(trainBatchNameList), imageHeight, imageWidth, 1])
                   for img in range(int(len(trainBatchNameList))):
                      mask[img,:trainHeights[img],:trainBatchSeqLengths[img],0]=np.ones([trainHeights[img],trainBatchSeqLengths[img]])
                   feed[mask_ph]=mask
                  
                summary, _ , batchCost, totalCost = sess.run([merged, optimizer, batch_cost, cost], feed)
                
    
                batch_end = time.time()
                time_elapsed=floor(1000*(batch_end-epoch_start))/1000
                prev_percent=floor(10000*(batch+1)/n_train_batches)/100
                remaining_time=max([0,floor(1000*(100*(time_elapsed+eps)/(prev_percent+eps)-time_elapsed))/1000])
                print('Epoch '+str(epoch+step*num_epochs)+'. Computed '+str(len(trainBatchNameList))+' sequences in batch '+str(batch+1)+'/'+str(n_train_batches)+'. Cost Function: '+str(batchCost)+'.\nTime elapsed: '+seconds_to_days_hours_min_sec(time_elapsed) +'. Remaining time: ' +seconds_to_days_hours_min_sec(remaining_time) + '\n')
                print('['+int(prev_percent)*'|'+(100-int(prev_percent))*' '+'] '+str(prev_percent)+'%\n')
                
                
                
            epoch_end=time.time()
            epoch_duration=epoch_end-epoch_start
            
            train_writer.add_summary(summary, epoch+step*num_epochs)
            print('Epoch '+ str(epoch+step*num_epochs) +' completed in: '+seconds_to_days_hours_min_sec(epoch_duration, day_flag=False)+'\n'*2)
            log_indicator.write('\nEpoch '+ str(epoch+step*num_epochs) +' completed in: '+seconds_to_days_hours_min_sec(epoch_duration, day_flag=False)+'. Cost: '+str(totalCost)+'\n')
            log_indicator.flush()
        saver.save(sess,models_path+'model',global_step=step*num_epochs+epoch)
                   

def validation(epoch, network, batchSize,set_name,Set, imageHeight,imageWidth,labels, num_classes,log_indicator, models_path, valid_writer, AACHEN_init=False, AACHEN_h5_file=[], dataAugmentation = False):

	nameList, inputs, targetList, seqLengths, heights, transcriptionList, transcriptionsLenList = Set

	SetSize=len(nameList)
	n_batches=ceil(SetSize/batchSize)

	nameList_copy, inputs_copy, targetList_copy, seqLengths_copy, heights_copy, transcriptionList_copy, transcriptionsLenList_copy=list(nameList), list(inputs), list(targetList), list(seqLengths), list(heights), list(transcriptionList), list(transcriptionsLenList)

	if dataAugmentation:
		   inputs_copy = pack_images(inputs_copy, imageHeight, imageWidth)

	setTotalChars= np.sum(transcriptionsLenList)          

	EDnorm=0
	EDabs=0
	totalCost=0
	list_trans=[]
	list_reco=[]
	graph, saver, inputs_mask_ph, seq_len_ph, targets_ph, targets_len_ph, learning_rate_ph, n_batches_ph, setTotalChars_ph, previousEDabs_ph, previousEDnorm_ph, previousCost_ph, optimizer, batch_cost, cost, errors, ED, predictions, merged = network.create(imageHeight,imageWidth, num_classes, True)

	if type(inputs_mask_ph) == list:
	  mask_ph=inputs_mask_ph[1]
	  inputs_ph=inputs_mask_ph[0]
	else:
	  inputs_ph=inputs_mask_ph
		
	if type(saver) == list:
	  saver=saver[0]

	with tf.Session(graph=graph) as sess:
		
		if AACHEN_init:
			text=('\nInitializing weights from AACHEN framework\n')
			print(text)
			log_indicator.write(text)
			init=tf.global_variables_initializer()
			
			feed_dict=initialize_from_AACHEN(graph, AACHEN_h5_file, log_indicator)
			sess.run(init, feed_dict=feed_dict)
		
		else:
			saver.restore(sess=sess,save_path=tf.train.latest_checkpoint(models_path))
		
		valid_start=time.time()
		prev_percent=-1
		
		text='\n'*4+"Muestras epoch "+ str(epoch)+ " in "+set_name+" set.\n"
		print(text)
		log_indicator.write(text)
		
		word_errors = 0
		num_words = 0

		for batch in range(int(n_batches)):
			BatchNameList, BatchInputs, BatchTargetSparse, BatchSeqLengths, BatchHeights, BatchTranscriptions, BatchTransLen=get_batch(batchSize, nameList_copy, inputs_copy, targetList_copy, seqLengths_copy, heights_copy, transcriptionList_copy, transcriptionsLenList_copy)

			feed = {inputs_ph: BatchInputs,
					targets_ph: BatchTargetSparse,
					targets_len_ph: BatchTransLen,
					seq_len_ph: BatchSeqLengths,
					n_batches_ph: n_batches,
					setTotalChars_ph: setTotalChars,
					previousEDabs_ph: EDabs,
					previousEDnorm_ph: EDnorm,
					previousCost_ph: totalCost}
			
			if type(inputs_mask_ph) == list:
				   mask=np.zeros([len(BatchNameList), imageHeight, imageWidth, 1])
				   for img in range(int(len(BatchNameList))):
					  mask[img,:BatchHeights[img],:BatchSeqLengths[img],0]=np.ones([BatchHeights[img],BatchSeqLengths[img]])
				   feed[mask_ph]=mask
			
			summary, batchCost, totalCost, [EDnorm, EDabs], BatchOutpusSparse, errors_output = sess.run([merged, batch_cost, cost, ED, predictions[0], errors], feed)
			
			
			BatchOutput=sess.run(tf.sparse_tensor_to_dense(tf.SparseTensor(BatchOutpusSparse.indices, BatchOutpusSparse.values, BatchOutpusSparse.dense_shape), default_value=num_classes))
			labels[num_classes]=' '
			for ind in range(int(len(BatchNameList))):
				obtained_transcription=' '.join(list(map(labels.get, list(BatchOutput[ind])))).strip()
				obtained_transcription1=''.join(list(map(labels.get, list(BatchOutput[ind])))).strip()
				text=str('| Name:').ljust(10)+str(BatchNameList[ind]).rjust(15)+' | '+str("Target:").ljust(10)+''.join(BatchTranscriptions[ind]).rjust(100)+" |\n"+str('| Errors: ').ljust(10)+str(errors_output[ind]).rjust(15)+' | '+str("Output:").ljust(10)+str(obtained_transcription).rjust(100)+' |\n'+'-'*88+'\n'
				print(text)
				tr=''.join(BatchTranscriptions[ind])
				
				reco=obtained_transcription1
				reco=reco.replace('sp',' ')
				tr=tr.replace('sp',' ')
				
				list_trans.append(tr +'\n')
				list_reco.append(reco +'\n')
				
				
				
				
				log_indicator.write(text)
				BatchTranscriptions[ind]=BatchTranscriptions[ind] +' sp'
				BatchTranscriptions[ind]=BatchTranscriptions[ind].replace('sp sp sp','sp')
				BatchTranscriptions[ind]=BatchTranscriptions[ind].replace('sp sp','sp')				 
				obtained_transcription=obtained_transcription + ' sp'
				obtained_transcription=obtained_transcription.replace('sp sp sp','sp')
				obtained_transcription=obtained_transcription.replace('sp sp','sp')
				#########################word error rate (sp symbol for splittttttttttttttttt)
				w1=levenshtein(''.join(BatchTranscriptions[ind].split()).split('sp'), ''.join(obtained_transcription.split()).split('sp'))
				w2=len(''.join(BatchTranscriptions[ind].split()).split('sp'))
				word_errors += levenshtein(''.join(BatchTranscriptions[ind].split()).split('sp'), ''.join(obtained_transcription.split()).split('sp'))
				num_words += len(''.join(BatchTranscriptions[ind].split()).split('sp'))
				print(word_errors)
				print(num_words)
				WERp =float(w1/w2)
				print('WERp%= ', str(WERp*100))
			batch_end = time.time()
			time_elapsed=floor(1000*(batch_end-valid_start))/1000
			prev_percent=floor(10000*(batch+1)/n_batches)/100
			remaining_time=floor(1000*(100*(time_elapsed+eps)/(prev_percent+eps)-time_elapsed))/1000
			print('Epoch '+str(epoch)+'. Evaluated '+str(len(BatchNameList))+' sequences in batch '+str(batch+1)+'/'+str(n_batches)+'. Cost Function: '+str(batchCost)+'.\nTime elapsed: '+seconds_to_days_hours_min_sec(time_elapsed) +'. Remaining time: ' +seconds_to_days_hours_min_sec(remaining_time) + '\n')
			print('['+int(prev_percent)*'|'+(100-int(prev_percent))*' '+'] '+str(prev_percent)+'%\n')
		

		WER =float(word_errors/num_words)     
		
		valid_writer.add_summary(summary, epoch)
		print('Total WER%= ', str(WER*100))
		##############save to files
		file1 = codecs.open('trans.txt', "w", "utf-8")
		file1.writelines(list_trans)
		file1.close()
		file2 = codecs.open('reco.txt', "w", "utf-8")
		file2.writelines(list_reco)
		file2.close()
		print_valid_results(epoch, set_name, SetSize, totalCost, [EDnorm, EDabs], WER, log_indicator)
		return totalCost

def transfer(epoch, network, imageHeight, imageWidth, num_classes, log_indicator, original_models_path, new_models_path, train_writer):

       graph, [saver, transfer_saver], inputs_mask_ph, seq_len_ph, targets_ph, targets_len_ph, learning_rate_ph, n_batches_ph, setTotalChars_ph, previousEDabs_ph, previousEDnorm_ph, previousCost_ph, optimizer, batch_cost, cost, errors, ED, predictions, merged = network.create(imageHeight,imageWidth, num_classes, False)

       with tf.Session(graph=graph) as sess:
            train_writer.add_graph(sess.graph) 
            text=('\nTransfering weights from {}\n'.format(original_models_path))
            print(text)
            log_indicator.write(text)
            tf.global_variables_initializer().run()
            transfer_saver.restore(sess=sess,save_path=tf.train.latest_checkpoint(original_models_path))
            saver.save(sess,new_models_path+'model', global_step=epoch)

def validation_logsfff(epoch, network, batchSize, set_name, Set, imageHeight, imageWidth, labels, num_classes, log_indicator, models_path, valid_writer,log_post_path, AACHEN_init=False, AACHEN_h5_file=[], dataAugmentation=False):
    
    nameList, inputs, targetList, seqLengths, heights, transcriptionList, transcriptionsLenList = Set

    SetSize = len(nameList)
    n_batches = 1#ceil(SetSize / batchSize)
    
    nameList_copy, inputs_copy, targetList_copy, seqLengths_copy, heights_copy, transcriptionList_copy, transcriptionsLenList_copy = list(nameList), list(inputs), list(targetList), list(seqLengths), list(heights), list(transcriptionList), list(transcriptionsLenList)
    

    
    graph, saver, inputs_mask_ph, seq_len_ph, targets_ph, targets_len_ph, learning_rate_ph, n_batches_ph, setTotalChars_ph, previousEDabs_ph, previousEDnorm_ph, previousCost_ph, optimizer, batch_cost, cost, errors, ED, predictions, merged = network.create(imageHeight, imageWidth, num_classes, True)
    

        
    if type(saver) == list:
      saver = saver[0]

    with tf.Session(graph=graph) as sess:
        
        if AACHEN_init:
            text = ('\nInitializing weights from AACHEN framework\n')
            print(text)
            log_indicator.write(text)
            init = tf.global_variables_initializer()
            
            feed_dict = initialize_from_AACHEN(graph, AACHEN_h5_file, log_indicator)
            sess.run(init, feed_dict=feed_dict)
        
        else:
            saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(models_path))
        
        valid_start = time.time()
        prev_percent = -1
        
        text = '\n' * 4 + "Samples at epoch " + str(epoch) + " in " + set_name + " set.\n"
        print(text)

        for batch in range(int(n_batches)):
			
			BatchNameList, BatchInputs, BatchTargetSparse, BatchSeqLengths, BatchHeights, BatchTranscriptions, BatchTransLen = get_batch(batchSize, nameList_copy, inputs_copy, targetList_copy, seqLengths_copy, heights_copy, transcriptionList_copy, transcriptionsLenList_copy)
			feed = {inputs_ph: BatchInputs,
					targets_ph: BatchTargetSparse,
					targets_len_ph: BatchTransLen,
					seq_len_ph: BatchSeqLengths,
					n_batches_ph: n_batches,
					setTotalChars_ph: setTotalChars,
					previousEDabs_ph: EDabs,
					previousEDnorm_ph: EDnorm,
					previousCost_ph: totalCost}
			
			if type(inputs_mask_ph) == list:
				mask = np.zeros(
					[len(BatchNameList), imageHeight, imageWidth, 1])
				for img in range(int(len(BatchNameList))):
					mask[img, :BatchHeights[img], :BatchSeqLengths[img], 0] = np.ones(
						[BatchHeights[img], BatchSeqLengths[img]])
				feed[mask_ph] = mask
			
			summary, batchCost, totalCost, [EDnorm, EDabs], BatchOutpusSparse, Logits, errors_output = sess.run(
				[merged, batch_cost, cost, ED, predictions[0], logits, errors], feed)
			end_load = time.time()
			toe = end_load - start_load
			#print('Batches is loaded in ' + seconds_to_days_hours_min_sec(toe))

			sLen, _, _ = np.shape(Logits)
			
			logs = ''
			print BatchNameList
			name=BatchNameList[0]
			filename ='/home/ahmed/Desktop/sana/RCNN/HTRTF/Projects/IFN-ENIT-WriterDep-Augmented-Comparison/decoding/nnet.ark'
			file = codecs.open(filename, "a", "utf-8")
			file.write(name + " [\n")
		
			########compute probss
			for seqn in range(0, BatchSeqLengths):
				seq = BatchInputs[seqn][0]
				file.write(str(seq[num_classes - 1]))
				file.write(" ")
				for c in range(0, num_classes - 1):
					val = seq[c]
					file.write(str(val))
					file.write(" ")
				file.write("\n")
			file.close
			# #reopen file and add "]"
			write(filename, "]")
	



def validation_logs(epoch, network, batchSize, set_name, Set, imageHeight, imageWidth, labels, num_classes, log_indicator, models_path, valid_writer,log_post_path, AACHEN_init=False, AACHEN_h5_file=[], dataAugmentation=False):

	list_decoded = []
	vocabReal = [line.strip() for line in open('/home/ahmed/Desktop/sana/RCNN/HTRTF/Projects/IFN-ENIT-WriterDep-Augmented-Comparison/Set/CHAR_LIST_new', 'r')]
	list_decoded_file = log_post_path
	nameList, inputs, targetList, seqLengths, heights, transcriptionList, transcriptionsLenList = Set

	SetSize = len(nameList)
	n_batches = ceil(SetSize / batchSize)
	print('stage 1')
	nameList_copy, inputs_copy, targetList_copy, seqLengths_copy, heights_copy, transcriptionList_copy, transcriptionsLenList_copy = list(
		nameList), list(inputs), list(targetList), list(seqLengths), list(heights), list(transcriptionList), list(transcriptionsLenList)

	if dataAugmentation:
		inputs_copy = pack_images(inputs_copy, imageHeight, imageWidth)

	setTotalChars = np.sum(transcriptionsLenList)
	print(seqLengths)
	EDnorm = 0
	EDabs = 0
	totalCost = 0
	logprobs = []
	graph, saver, inputs_mask_ph, seq_len_ph, targets_ph, targets_len_ph, learning_rate_ph, n_batches_ph, setTotalChars_ph, previousEDabs_ph, previousEDnorm_ph, previousCost_ph, optimizer, batch_cost, cost, errors, ED, predictions, logits, merged = network.create(
		imageHeight, imageWidth, num_classes, True)

	if type(inputs_mask_ph) == list:
		mask_ph = inputs_mask_ph[1]
		inputs_ph = inputs_mask_ph[0]
	else:
		inputs_ph = inputs_mask_ph

	if type(saver) == list:
		saver = saver[0]

	with tf.Session(graph=graph) as sess:
		
		print('Computing Probabilities\n')
 
		saver.restore(
				sess=sess, save_path=tf.train.latest_checkpoint(models_path))

		valid_start = time.time()
		prev_percent = -1

		text = '\n' * 4 + "Evaluation on the latest checkpoint in " + set_name + " set.\n"
		print(text)
		log_indicator.write(text)

		word_errors = 0
		num_words = 0
		i = 0
		for batch in range(int(n_batches)):
			BatchNameList, BatchInputs, BatchTargetSparse, BatchSeqLengths, BatchHeights, BatchTranscriptions, BatchTransLen = get_batch(
				batchSize, nameList_copy, inputs_copy, targetList_copy, seqLengths_copy, heights_copy, transcriptionList_copy, transcriptionsLenList_copy)
			print(BatchSeqLengths)
			feed = {inputs_ph: BatchInputs,
					targets_ph: BatchTargetSparse,
					targets_len_ph: BatchTransLen,
					seq_len_ph: BatchSeqLengths,
					n_batches_ph: n_batches,
					setTotalChars_ph: setTotalChars,
					previousEDabs_ph: EDabs,
					previousEDnorm_ph: EDnorm,
					previousCost_ph: totalCost}
			print('stage 2')

			summary, batchCost, totalCost, [EDnorm, EDabs], BatchOutpusSparse, Logits, errors_output = sess.run(
				[merged, batch_cost, cost, ED, predictions[0], logits, errors], feed)
			sLen, _, _ = np.shape(Logits)
			print(sLen)
			print(np.shape(Logits))
			logs = ''
			
			
			#imageName = imageName[:9]
			dir_decode = '/home/ahmed/Desktop/sana/RCNN/HTRTF/Projects/IFN-ENIT-WriterDep-Augmented-Comparison/decoding/'
			list_decoded_file = dir_decode + '/nnet.ark'
			# create dir
			if (os.path.exists(dir_decode) == False): os.makedirs(dir_decode)
			
			
			 
			for i in range(0, len(BatchNameList)):
				print(BatchNameList[i])
				logs = ''
				# compute probss
				for seqn in range(0, sLen):
					# print seqn
					seq = Logits[seqn][i]
					logs = logs + str(seq[num_classes - 1])
					logs = logs + ";"
					for c in range(0, num_classes - 1):
						val = seq[c]
						logs = logs + str(val)
						logs = logs + ";"
					logs = logs + "\n"
			
					
				ark = createArk(logs, vocabReal)
				ark = BatchNameList[i] + ' [\n' + ark
				logprobs.append(ark + '\n')

	 
	with codecs.open(list_decoded_file, 'w', 'utf-8') as f1:
		f1.writelines(logprobs)  # save to file
			
			