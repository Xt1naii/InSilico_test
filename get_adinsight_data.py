from concurrent.futures import ThreadPoolExecutor
import logging

import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_drug(url):
    response = requests.get(url).text
    soup = BeautifulSoup(response, 'lxml')
    drug_dict = {}
    logging.warning(f"Parsing {url}")

    if soup.find(class_="page__error"):
        return None

    drug_name = soup.find(id="drugNameID").text.strip()

    try:
        alternative_names = soup.find(class_="document__alt-name").text.strip() \
                            .replace("Alternative Names:", " ").strip() \
                            .replace("; ", ";").split(";")
    except:
        alternative_names = None

    try:
        drug_classes = soup.find(id="at-a-glance_class") \
                       .find(class_="data-list__property-value").text.strip() \
                       .replace("; ", ";").split(";")
    except:
        drug_classes = None

    try:
        drug_mechanism = soup.find(id="at-a-glance_mechanismOfAction") \
                         .find(class_="data-list__property-value").text.strip() \
                         .replace("; ", ";").split(";")
    except:
        drug_mechanism = None

    try:
        drug_orphan_st = soup.find(id="at-a-glance_orphanStatus") \
                         .find(class_="data-list__property-value").text.strip()
    except:
        drug_orphan_st = None

    try:
        new_molecular_entity = soup.find(id="at-a-glance_newMolecularEntity") \
                               .find(class_="data-list__property-value").text.strip()
    except:
        new_molecular_entity = None

    try:
        drug_dev_phase = soup.find(class_="data-list__content data-list__content--highest-dev-phases") \
                               .find(class_="data-list__property-key").text.strip()
        drug_dev_illness = soup.find(class_="data-list__content data-list__content--highest-dev-phases")\
                               .find(class_="data-list__property-value").text.strip().split("; ")
    except:
        drug_dev_phase = None
        drug_dev_illness = None

    try:
        recent_events = [x.text.strip() for x in soup.find(class_="data-list__content data-list__content--most-recent-events").find_all(class_="data-list__property-value property-value--event-details")]
    except:
        recent_events = None

    drug_dict['drug_name'] = drug_name
    drug_dict['alternative_names'] = alternative_names
    drug_dict['drug_class'] = drug_classes
    drug_dict['mechanism'] = drug_mechanism
    drug_dict['orphan_drug_status'] = drug_orphan_st
    drug_dict['new_molec_ent'] = new_molecular_entity
    drug_dict['dev_phase'] = drug_dev_phase
    drug_dict['dev_illness'] = drug_dev_illness
    drug_dict['recent_events'] = recent_events
    drug_dict['url'] = url

    return drug_dict

def get_adisinsight_data():
    urls = [f'https://adisinsight.springer.com/drugs/{800000000 + i}' for i in range(0, 74000)]

    with ThreadPoolExecutor() as executor:
        result = list(executor.map(scrape_drug, urls))
    result = [x for x in result if x]
    result = pd.DataFrame(result)
    result.to_parquet('./data/scrape_result.parquet')

if __name__ == '__main__':
    get_adisinsight_data()
