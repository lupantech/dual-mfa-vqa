import json
import os

## preprocess question json file, {question_id : question, image_name}
# target_path = '/Users/dbxiaolu/Desktop/leiji/vqa./'
input_json = '../data_train-val_test-dev_2k/vqa_raw_test.json'
output_question_json = 'test-dev_prepro_questions.json'
out = {}

question_list = json.load(open(input_json, 'r'))
for i, ques in enumerate(question_list):
    # if i < 1: print ques, '\n'
    # {u'ques_id': 4195880, u'question': u'Are the dogs tied?', u'img_path': u'test-dev2015/COCO_test2015_000000419588.jpg'}    
    que_id = ques['ques_id']
    image_ques = {'image_name':os.path.basename(ques['img_path']), 'question':ques['question']}
    out[que_id] = image_ques

print 'length of output question list = ', len(out)
print 4195880, out[4195880]
print 4195881, out[4195881]
# { 4195880:{'question': u'Are the dogs tied?', 'image_name': u'COCO_test2015_000000419588.jpg'}, 
#   4195881 {'question': u'Is this a car show?', 'image_name': u'COCO_test2015_000000419588.jpg'}, }

json.dump(out, open(output_question_json, 'w'))
print '\n wrote ', output_question_json