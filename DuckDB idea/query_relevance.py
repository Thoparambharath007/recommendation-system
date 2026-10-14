"""Reject clearly unrelated or unrecognized input before asking for SQL."""
import re
from request_policy import COLOR_PATTERN

SHIRT_WORDS=re.compile(r'\b(?:shirts?|shrt|shrit|blouses?|polos?|button[ -]down|button[ -]up)\b',re.I)
DETAIL_WORDS=re.compile(r'\b(?:comfortable|comfy|comfort|office|workplace|business|formal|casual|cotton|linen|polyester|nylon|silk|wool|rayon|viscose|spandex|elastane|modal|bamboo|breathable|soft|stretch|lightweight|plain|solid|striped|stripes|checked|checkered|plaid|floral|polka|camouflage|collar|collared|slim|regular|relaxed|loose|fit|sleeves?|pockets?|sizes?|cheapest|ratings?|rated|reviews?|popular|budget)\b',re.I)
OTHER_GOODS=re.compile(r'\b(?:shoes?|sneakers?|boots?|pants?|trousers?|jeans|jackets?|blazers?|hoodies?|dresses|skirts?|watches|phones?|laptops?|cars?|t[ -]?shirts?)\b',re.I)
OTHER_TASKS=re.compile(r'\b(?:weather|forecast|recipes?|flight|hotels?|bitcoin|stock market|football score|cricket score|tell (?:me )?a joke|write (?:a |some )?code|python code|solve (?:this |a )?(?:math|equation)|translate|politics)\b',re.I)

def clarification_reason(query,brands=()):
    if not query.strip():return None
    text=query.strip().casefold()
    if any(text==brand.strip().casefold() for brand in brands):return None
    if OTHER_TASKS.search(query):return 'This search is for shirts. Try a color, style, size or budget.'
    if OTHER_GOODS.search(query) and not SHIRT_WORDS.search(query):
        return 'This catalog contains shirts. Try describing the shirt you want.'
    if COLOR_PATTERN.search(query) or SHIRT_WORDS.search(query) or DETAIL_WORDS.search(query):return None
    if re.fullmatch(r'(?:XXS|XS|S|M|L|XL|XXL|[3-8]XL)(?: Tall)?',query.strip()):return None
    if re.search(r'\b(?:under|below|up to|at most|above|over)\s*(?:₹|\$|rs\.?|inr|usd)?\s*\d',query,re.I):return None
    for brand in brands:
        name=brand.strip().casefold()
        if name and re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)',text):return None
    return 'We couldn’t understand that search. Describe a shirt, or choose a brand from the filters.'

SUGGESTIONS=['Comfortable white shirt for office','Black shirt under ₹1,200','Blue cotton shirt size M']
