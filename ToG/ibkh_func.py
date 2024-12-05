from prompt_list import *
import json
import openai
import re
import time
from utils import *
import os
from tqdm import tqdm
import pickle


dict_filenames = ['tail2rels_dict.pkl', 'head2tails_dict.pkl', 'tail@rel2heads_dict.pkl', 'head@rel2tails_dict.pkl', 'id2label.pkl', 'rel@tail2heads_dict.pkl', 'tail2heads_dict.pkl', 'label2id.pkl', 'head2rels_dict.pkl']

script_dir = os.path.dirname(os.path.abspath(__file__))
dict_dir = os.path.join(script_dir, '..', 'ibkh_data')

dicts_by_filename = {}

for filename in tqdm(dict_filenames, desc="Loading dicts"):
    file_path = os.path.join(dict_dir, filename)
    with open(file_path, 'rb') as file:
        dicts_by_filename[filename] = pickle.load(file)

json_file_path = os.path.join(script_dir, '..', 'ibkh_data', 'relation_dict.json')

with open(json_file_path, 'r') as file:
    rels2labels = json.load(file)

labels2rels = {rels2labels[k]:k for k in rels2labels}


def id2entity_name_or_type(entity_id):
    """
    Returns the label of the input entity ID
    """
    try:
        res = dicts_by_filename['id2label.pkl'][entity_id]
    except KeyError as e:
        print(f"Keyerror: {e}")
        res = []
    return res


def abandon_rels(relation):
    return False


def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    """
    Returns prompt string for relation extraction with scoring
    """
    return extract_relation_prompt % (args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "
        

def construct_entity_score_prompt(question, relation, entity_candidates):
    """
    Returns prompt string for scoring entities
    """
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '


def clean_relations(string, entity_id, head_relations):
    """
    Parses LLM output and returns list of relations
    """
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args):
    """
    Returns list of relations for given entity ID
    """
    try:
        head_relations = dicts_by_filename['head2rels_dict.pkl'][entity_id]
    except KeyError as e:
        print(f"Keyerror: {e}")
        head_relations = []
    try:
        tail_relations = dicts_by_filename['tail2rels_dict.pkl'][entity_id]
    except KeyError as e:
        print(f"Keyerror: {e}")
        tail_relations = []
    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations+tail_relations
    total_relations.sort()  # make sure the order in prompt is always equal
    
    mapped_total_relations = [rels2labels[r] if r in rels2labels else r for r in total_relations]
    prompt = construct_relation_prune_prompt(question, entity_name, mapped_total_relations, args)

    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
    flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations) 

    if flag:
        return retrieve_relations_with_scores
    else:
        return [] # format error or too small max_length
    

def entity_search(entity, relation, head=True):
    """
    Returns list of tail entities for given entity ID and relation
    """
    if head:
        try:
            if relation in labels2rels:
                entities = dicts_by_filename['head@rel2tails_dict.pkl'][f'{entity}@{labels2rels[relation]}']
            else:
                entities = dicts_by_filename['head@rel2tails_dict.pkl'][f'{entity}@{relation}']
        except KeyError as e:
            print(f"Keyerror: {e}")
            entities = []
    else:
        try:
            if relation in labels2rels:
                entities = dicts_by_filename['tail@rel2heads_dict.pkl'][f'{entity}@{labels2rels[relation]}']
            else:
                entities = dicts_by_filename['tail@rel2heads_dict.pkl'][f'{entity}@{relation}']
        except KeyError as e:
            print(f"Keyerror: {e}")
            entities = []
    return entities


def entity_score(question, entity_candidates_id, score, relation, args):
    """
    Returns list of scores of all candidate entities
    """
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]
    if all_unknown_entity(entity_candidates):
        return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    entity_candidates = del_unknown_entity(entity_candidates)
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)
    if args.prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        return [float(x) * score for x in clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id
    else:
        print("Unsupported prune tool")

    
def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


def half_stop(question, cluster_chain_of_entities, depth, args):
    print("No new knowledge added during search depth %d, stop searching." % depth)
    answer = generate_answer(question, cluster_chain_of_entities, args)
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset)


def generate_answer(question, cluster_chain_of_entities, args):
    """
    Returns LLM answer output given reasoning chain
    """ 
    prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return result


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args):
    """
    Returns pruned lists of reasoning chains, entities and relations based on the width parameter
    """
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    tops = [id2entity_name_or_type(entity_id) for entity_id in tops]
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads


def reasoning(question, cluster_chain_of_entities, args):
    """
    Returns True if given reasoning chain is sufficient to answer the question, False otherwise 
    """
    prompt = prompt_evaluate + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    
    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response