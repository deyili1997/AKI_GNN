import requests

def get_rxcui_list(drug_name):
    """
    Fetches the list of RXCUI values for the given drug name.

    Args:
        drug_name (str): The name of the drug to query.

    Returns:
        list: A list of RXCUI values for the specified drug name.
    """
    base_url = "https://rxnav.nlm.nih.gov/REST/drugs.json"
    
    try:
        response = requests.get(base_url, params={"name": drug_name})
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
    except requests.RequestException as e:
        print(f"Failed to fetch RXCUI for {drug_name}. Error: {e}")
        return []
    
    data = response.json()
    
    if 'conceptGroup' not in list(data["drugGroup"].keys()):
        print("No RXCUI found for", drug_name)
        return []
    
    scd_group = next((group for group in data["drugGroup"]["conceptGroup"] if group.get("tty") == "SCD"), None)

    if scd_group and "conceptProperties" in scd_group:
        scd_rxcuis = [concept["rxcui"] for concept in scd_group["conceptProperties"]]
        return scd_rxcuis
    else:
        print("No RXCUI found for", drug_name)
        return []
    