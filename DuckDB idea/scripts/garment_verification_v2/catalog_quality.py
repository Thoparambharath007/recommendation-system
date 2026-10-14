"""Conservative product-type exclusions and exact saved palette evidence."""
import json
import math
import re

SHIRT = re.compile(r'\b(?:shirts?|polos?|blouses?)\b',re.I)
OUTERWEAR = re.compile(r'\b(?:blazers?|jackets?|coats?|suits?|hoodies?|sweaters?|sweatshirts?|vests?)\b',re.I)


def non_shirt_reason(title):
    """Exclude a different primary garment, not a color or styling mention."""
    text=str(title or '')
    # These listings have a shirt-shaped top but are one-piece garments.
    # Preserve incidental styling text ('shirt to wear with a bodysuit').
    primary=re.split(r'\b(?:to wear with|to wear under|pair with|paired with|for wearing with)\b',text,flags=re.I)[0]
    one_piece=re.search(r'\b(?:body[ -]?suits?|leotards?|rompers?|jumpsuits?)\b',primary,re.I)
    if one_piece:
        return 'One-piece garment: '+one_piece.group().lower()
    # A description such as 'jacket style shirt' is not a jacket listing.
    text=re.sub(r'\b(?:jacket|coat)[ -]style\b',' ',text,flags=re.I)
    text=re.sub(r'\b(?:coat[ -]front|mid[ -]coat)\b',' ',text,flags=re.I)
    shirt=SHIRT.search(text)
    garment=OUTERWEAR.search(text)
    if garment and shirt is None:
        return 'Primary title garment is '+garment.group().lower()
    if garment and shirt and garment.start()<shirt.start():
        before=text[:shirt.start()]
        # One word can be a brand (Pepe Jeans), line name (Arrow Blazer),
        # or hybrid shirt description. Require corroborating outerwear wording.
        outer=list(re.finditer(r'\b(?:blazers?|jackets?|coats?|suits?)\b',before,re.I))
        clear_lead=re.search(r'^\s*(?:jackets?|blazers?|coats?|suits?)\s+for\s+men\b',text,re.I)
        if len(outer)>=2 or clear_lead:
            return 'Primary outerwear title: '+', '.join(m.group().lower() for m in outer)
    return ''


def palette_dominance(raw):
    """Never merge different shades or infer percentages from tag order."""
    try:
        values=json.loads(raw) if isinstance(raw,str) else raw
        if not isinstance(values,list) or not values:return None,None
        shares={}
        for item in values:
            name=item['name'].strip().lower()
            share=item['share']
            if not isinstance(share,(int,float)) or isinstance(share,bool) or not math.isfinite(share) or not 0<=share<=1:
                return None,None
            if name in shares:return None,None
            shares[name]=float(share)
        if sum(shares.values())>1.001:return None,None
        highest=max(shares.values())
        leaders=[name for name,share in shares.items() if share==highest]
        if highest==0 or len(leaders)!=1:return None,None
        return leaders[0],highest
    except (ValueError,TypeError,KeyError,AttributeError):
        return None,None
