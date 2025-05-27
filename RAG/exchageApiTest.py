import requests

def fetch_top_qa(query,
                 site='music.stackexchange',
                 num_questions=5,
                 num_answers=2,
                 key=None):
    # 1. Search for the first `num_questions` matching questions
    q_params = {
        'site': site,
        'q': query,
        'pagesize': num_questions,
        'sort': 'relevance',
        'order': 'desc',
    }
    if key:
        q_params['key'] = key  # optional API key to raise quotas :contentReference[oaicite:0]{index=0}

    q_resp = requests.get('https://api.stackexchange.com/2.3/search/advanced',
                          params=q_params)
    q_resp.raise_for_status()
    questions = q_resp.json().get('items', [])

    results = []
    for q in questions:
        qid    = q['question_id']
        title  = q['title']
        link   = q['link']

        # 2. Fetch the top `num_answers` by votes for each question
        a_params = {
            'site': site,
            'sort': 'votes',         # sort answers by score :contentReference[oaicite:1]{index=1}
            'order': 'desc',
            'pagesize': num_answers,
            'filter': 'withbody'     # include the full answer body :contentReference[oaicite:2]{index=2}
        }
        if key:
            a_params['key'] = key

        a_resp = requests.get(
            f'https://api.stackexchange.com/2.3/questions/{qid}/answers',
            params=a_params
        )
        a_resp.raise_for_status()
        answers = a_resp.json().get('items', [])

        # 3. Extract answer bodies
        top_answers = [ans['body'] for ans in answers]

        # 4. Structure for RAG: source link, question title, and top answer texts
        results.append({
            'source':      link,
            'question':    title,
            'top_answers': top_answers
        })

    return results

if __name__ == '__main__':
    query = "Synthwave"
    for item in fetch_top_qa(query, num_questions=10, num_answers=5):
        print(f"- source: {item['source']}")
        print(f"  Question: {item['question']}")
        print(f"  Top Answers:")
        for i, ans in enumerate(item['top_answers'], 1):
            # trim or sanitize HTML as needed for your RAG system
            print(f"    {i}. {ans}\n")
        print()
