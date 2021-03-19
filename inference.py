import time
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from utils import AverageMeter
from utils import AverageMeter, calculate_accuracy


def get_video_results(outputs, class_names, output_topk):
    sorted_scores, locs = torch.topk(outputs,
                                     k=min(output_topk, len(class_names)))

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return video_results


def inference(data_loader, model, result_path, class_names, no_average,
              output_topk):
    print('inference')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()
    results = {'results': defaultdict(list)}

    end_time = time.time() 
    

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            
            video_ids, segments = zip(*targets)
            # print("INPUTS", inputs)
            # print("INPUTS size", len(inputs))
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1).cpu()
            acc = calculate_accuracy(outputs, targets)
            print(len(inputs))
            accuracies.update(acc, inputs.size(0))

            for j in range(outputs.size(0)):
                results['results'][video_ids[j]].append({
                    'segment': segments[j],
                    'output': outputs[j]
                })

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time),
                      acc=accuracies)

    inference_results = {'results': {}}
    if not no_average:
        for video_id, video_results in results['results'].items():
            video_outputs = [
                segment_result['output'] for segment_result in video_results
            ]
            video_outputs = torch.stack(video_outputs)
            average_scores = torch.mean(video_outputs, dim=0)
            inference_results['results'][video_id] = get_video_results(
                average_scores, class_names, output_topk)
    else:
        for video_id, video_results in results['results'].items():
            inference_results['results'][video_id] = []
            for segment_result in video_results:
                segment = segment_result['segment']
                result = get_video_results(segment_result['output'],
                                           class_names, output_topk)
                inference_results['results'][video_id].append({
                    'segment': segment,
                    'result': result
                })
    inference_results['accuracy'] = accuracies.avg
    
    with result_path.open('w') as f:
        json.dump(inference_results, f)
    
    return inference_results
