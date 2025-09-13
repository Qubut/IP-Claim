import pytest

from utils.entity_extractor import extract_entities


@pytest.fixture(scope='module')
def small_text():
    return 'Apple Inc. is an American company. It designs consumer electronics.'


@pytest.fixture(scope='module')
def coref_text():
    return (
        'John Smith works at Google. He is a software engineer. '
        'The company hired him last year. Microsoft is his competitor.'
    )


@pytest.fixture(scope='module')
def long_text():
    return 'NASA was established in 1958. ' * 500  # ~10,000 characters


def test_small_text_processing(small_text):
    """Test processing of small text that doesn't require chunking."""
    entities = extract_entities(small_text)

    # Should contain at least 2 entities
    assert len(entities) >= 2

    # Verify Apple Inc. is recognized
    apple_entity = [e for e in entities if e[0] == 'Apple Inc.']
    assert len(apple_entity) > 0
    assert apple_entity[0][1] == 'ORG'  # Organization label

    # Verify consumer electronics is recognized
    electronics_entity = [e for e in entities if 'consumer electronics' in e[0]]
    assert len(electronics_entity) > 0


def test_coreference_resolution(coref_text):
    """Test coreference resolution across mentions."""
    entities = extract_entities(coref_text)

    # Extract all mentions of John
    john_mentions = [e for e in entities if e[0] == 'John Smith' or e[0] == 'He' or e[0] == 'him']
    assert len(john_mentions) >= 3

    # All should have the same label (PERSON)
    labels = {e[1] for e in john_mentions}
    assert len(labels) == 1
    assert 'PERSON' in labels

    # Verify company mentions
    google_mentions = [e for e in entities if 'Google' in e[0]]
    company_mentions = [e for e in entities if 'company' in e[0] and e[1] == 'ORG']
    assert len(google_mentions) > 0
    assert len(company_mentions) > 0

    # Microsoft should be recognized
    microsoft_mentions = [e for e in entities if 'Microsoft' in e[0]]
    assert len(microsoft_mentions) > 0
    assert microsoft_mentions[0][1] == 'ORG'


def test_large_text_processing(long_text):
    """Test chunk processing for large texts"""
    entities = extract_entities(long_text, chunk_size=2000)

    # Should find multiple NASA mentions
    nasa_mentions = [e for e in entities if 'NASA' in e[0]]
    assert len(nasa_mentions) >= 50

    # All should be recognized as ORG
    labels = {e[1] for e in nasa_mentions}
    assert len(labels) == 1
    assert 'ORG' in labels

    # Verify year mentions
    year_mentions = [e for e in entities if '1958' in e[0]]
    assert len(year_mentions) >= 50


def test_entity_boundaries():
    """Test correct entity boundary detection"""
    text = 'Dr. Jane Doe works at Massachusetts Institute of Technology (MIT) in Boston.'
    entities = extract_entities(text)

    # Verify full entity names are captured
    person = [e for e in entities if 'Jane Doe' in e[0]]
    org = [e for e in entities if 'Massachusetts Institute of Technology' in e[0]]
    acronym = [e for e in entities if 'MIT' in e[0]]
    location = [e for e in entities if 'Boston' in e[0]]

    assert len(person) > 0
    assert person[0][0] == 'Dr. Jane Doe'

    assert len(org) > 0
    assert org[0][0] == 'Massachusetts Institute of Technology'

    assert len(acronym) > 0
    assert acronym[0][0] == 'MIT'

    assert len(location) > 0
    assert location[0][0] == 'Boston'


def test_no_entities():
    """Test text with no entities"""
    text = 'This is a simple sentence without any named entities.'
    entities = extract_entities(text)
    assert len(entities) == 0


def test_special_characters():
    """Test text with special characters"""
    text = "Elon Musk's company SpaceX (spacex.com) launched a $100M project."
    entities = extract_entities(text)

    # Verify entities with special characters
    person = [e for e in entities if 'Elon Musk' in e[0]]
    company = [e for e in entities if 'SpaceX' in e[0]]
    website = [e for e in entities if 'spacex.com' in e[0]]
    project = [e for e in entities if '$100M' in e[0]]

    assert len(person) > 0
    assert person[0][1] == 'PERSON'

    assert len(company) > 0
    assert company[0][1] == 'ORG'

    # Depending on model, website might be recognized as ORG or not at all
    if website:
        assert website[0][1] in ['ORG', 'PRODUCT']

    # Monetary values might be recognized as MONEY
    if project:
        assert project[0][1] == 'MONEY'


def test_coref_label_consistency():
    """Test consistent labeling across coreference mentions"""
    text = (
        'Amazon announced a new product. The company said it would ship next month. '
        'Jeff Bezos founded it. He remains involved.'
    )

    entities = extract_entities(text)

    # Amazon and company should have same label
    amazon = [e for e in entities if e[0] == 'Amazon']
    company = [e for e in entities if e[0] == 'The company']

    assert len(amazon) > 0
    assert len(company) > 0
    assert amazon[0][1] == company[0][1] == 'ORG'

    # Jeff Bezos and He should have same label
    jeff = [e for e in entities if 'Jeff Bezos' in e[0]]
    he = [e for e in entities if e[0] == 'He']

    assert len(jeff) > 0
    assert len(he) > 0
    assert jeff[0][1] == he[0][1] == 'PERSON'
