import itertools

import numpy as np
import pandas as pd

TAIPOWER_TABLE = 'data/2017-201801.csv'
FOUNDATIONS_TABLE = 'preprocessed_2.csv'
TARGETS_TABLE = 'targets.tsv'

foundations_data = pd.read_csv(FOUNDATIONS_TABLE, index_col=0).drop_duplicates()
director_weights = foundations_data.groupby('name').count()


def preprocesss_taipower():
    taipower_full = pd.read_csv(TAIPOWER_TABLE)
    sliced = taipower_full[['補助縣市', '申請補(捐)助單位', '活動名稱', '核准補助金額(元)']]
    sliced.columns = ['place', 'org', 'title', 'amount']
    sliced = sliced.astype({'place': str, 'org': str, 'title': str, 'amount': int})
    # remove all '促協金' as they are too much of money and refer to external reference...
    return sliced[~sliced['title'].str.contains('促協金')]


taipower_data = preprocesss_taipower()
targets_data = pd.read_csv(TARGETS_TABLE, sep='\t')


all_congress_members = list(targets_data['民意代表姓名'].drop_duplicates())


def _transitive(root: pd.DataFrame, max_iter, min_weight=1, max_weight=None):
    current = root.drop_duplicates()

    generations = current[['name']].drop_duplicates()
    generations['gen'] = 0

    if max_weight:
        valid_weights = director_weights[director_weights['org'] <= max_weight]
    else:
        valid_weights = None

    # if min_weight > 1:
    #     valid_weights = director_weights[director_weights['org'] >= min_weight]
    # else:
    #     valid_weights = director_weights

    # print('orig orgs=%d, ppl=%d' % (current.shape[0], generations.shape[0]))

    for i in range(1, max_iter):
        allppl = foundations_data.merge(
            current[['org']].drop_duplicates(), on='org')[['name']].drop_duplicates()
        if valid_weights is not None:
            allppl = allppl.join(valid_weights, on='name', how='inner')[['name']]
        newppl = allppl.append(generations[['name']], ignore_index=True).drop_duplicates(keep=False)

        # stop if no new director is found
        if newppl.empty:
            break

        newppl['gen'] = i
        generations = generations.append(newppl, ignore_index=True)

        # find new orgs based on newppl
        current = foundations_data.merge(allppl, on='name')[['name', 'org']]
        # next_step = next_step.drop_duplicates()
        # if next_step.shape[0] == current.shape[0]:
        #     break
        # current = next_step
        # print('orgs=%d, ppl=%d' % (current[['org']].drop_duplicates().shape[0], generations.shape[0]))

    return current, generations


def compute_by_human_network(name, max_iter=6, max_weight=50):
    if max_iter < 1:
        raise ValueError('max_iter should be at least 1')

    start = targets_data[targets_data['民意代表姓名'].str.match(name)][['民意代表親屬姓名']].dropna()
    start.columns = ['name']
    start.append(pd.DataFrame([{'name': name}]))
    root = start.merge(foundations_data, on='name')[['name', 'org']]

    related_ppl_and_orgs, _ = _transitive(root, max_iter, max_weight)
    return related_ppl_and_orgs.merge(taipower_data, on='org')


def compute_by_org(org):
    return taipower_data.loc[taipower_data['org'] == org][['org', 'amount']].groupby(by='org').sum()


def most_valuable_director(n=100):
    money_per_org = taipower_data[['org', 'amount']].groupby(by='org').sum()
    money_directors = foundations_data.merge(money_per_org, left_on='org', right_index=True)
    money_per_director = money_directors[['name', 'amount']].groupby('name').sum().sort_values(by='amount', ascending=False)
    return money_per_director.head(n)


def demo(degree):
    total_amount = 0
    for member in all_congress_members:
        result = compute_by_human_network(member, degree)
        print('%s' % member)
        print('Total possible amount: %d' % result['amount'].sum())
        total_amount += result['amount'].sum()
        if not result.empty:
            print(result)

    print('TOTAL: ', total_amount)
