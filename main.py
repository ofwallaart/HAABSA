# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC

import tensorflow as tf
import lcrModel
import lcrModelInverse
import lcrModelAlt
import cabascModel
import svmModel
from OntologyReasoner import OntReasoner
from loadData import *

#import parameter configuration and data paths
from config import *

#import modules
import numpy as np
import sys

# main function
def main(_):
    loadData = False
    useOntology = False
    runCABASC = False
    runLCRROT = False
    runLCRROTINVERSE = False
    runLCRROTALT = True
    runSVM = False

    weightanalysis = False

    #determine if backupmethod is used
    if runCABASC or runLCRROT or runLCRROTALT or runLCRROTINVERSE or runSVM:
        backup = True
    else:
        backup = False
    
    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData)
    print(test_size)
    remaining_size = 250
    accuracyOnt = 0.87

    if useOntology == True:
        print('Starting Ontology Reasoner')
        Ontology = OntReasoner()
        #out of sample accuracy
        accuracyOnt, remaining_size = Ontology.run(backup,FLAGS.test_path, runSVM)
        #in sample accuracy
        Ontology = OntReasoner()
        accuracyInSampleOnt, remaining_size = Ontology.run(backup,FLAGS.train_path, runSVM)
        if runSVM == True:
            test = FLAGS.remaining_svm_test_path
        else:
            test = FLAGS.remaining_test_path
        print('train acc = {:.4f}, test acc={:.4f}, remaining size={}'.format(accuracyOnt, accuracyOnt, remaining_size))
    else:
        if runSVM == True:
            test = FLAGS.test_svm_path
        else:
            test = FLAGS.test_path

    # LCR-Rot model
    if runLCRROT == True:
        _, pred1, fw1, bw1, tl1, tr1, sent, target, true = lcrModel.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
        tf.reset_default_graph()

    # LCR-Rot-inv model
    if runLCRROTINVERSE == True:
        lcrModelInverse.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
        tf.reset_default_graph()

    # LCR-Rot-hop model
    if runLCRROTALT == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
        tf.reset_default_graph()

    # CABASC model
    if runCABASC == True:
        _, pred3, weights = cabascModel.main(FLAGS.train_path,test, accuracyOnt, test_size, remaining_size)
        if weightanalysis and runLCRROT and runLCRROTALT:
            outF= open('sentence_analysis.txt', "w")
            dif = np.subtract(pred3, pred1)
            for i, value in enumerate(pred3):
                if value == 1 and pred2[i] == 0:
                    sentleft, sentright = [], []
                    flag = True
                    for word in sent[i]:
                        if word == '$t$':
                            flag = False
                            continue
                        if flag:
                            sentleft.append(word)
                        else:
                            sentright.append(word)
                    print(i)
                    outF.write(str(i))
                    outF.write("\n")
                    outF.write('lcr pred: {}; CABASC pred: {}; lcralt pred: {}; true: {}'.format(pred1[i], pred3[i], pred2[i], true[i]))
                    outF.write("\n")
                    outF.write(";".join(sentleft))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in fw1[i][0]))
                    outF.write("\n")
                    outF.write(";".join(sentright))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in bw1[i][0]))
                    outF.write("\n")
                    outF.write(";".join(target[i]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in tl1[i][0]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in tr1[i][0]))
                    outF.write("\n")
                    outF.write(";".join(sentleft))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in fw2[i][0]))
                    outF.write("\n")
                    outF.write(";".join(sentright))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in bw2[i][0]))
                    outF.write("\n")
                    outF.write(";".join(target[i]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in tl2[i][0]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in tr2[i][0]))
                    outF.write("\n")
                    outF.write(";".join(sent[i]))
                    outF.write("\n")
                    outF.write(";".join(str(x) for x in weights[i][0]))
                    outF.write("\n")
            outF.close()

    # BoW model
    if runSVM == True:
        svmModel.main(FLAGS.train_svm_path,test, accuracyOnt, test_size, remaining_size)

    print('Finished program succesfully')

if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()
