/* Product data is rendered as text, never HTML supplied by source listings. */
(() => {
 const dialog=document.getElementById('product-dialog');
 const content=document.getElementById('product-content');
 const standalone=document.body.dataset.productPage==='true';
 let chosenSize=null;
 let pending=null;
 const node=(tag,value='',className='')=>{const element=document.createElement(tag);element.textContent=value;if(className)element.className=className;return element;};
 const safeURL=value=>{try{const u=new URL(value);return ['https:','http:'].includes(u.protocol)?u.href:null;}catch{return null;}};
 const valueOrUnknown=value=>value && value!=='unknown'?String(value).replaceAll('_',' '):'Not recorded';
 const link=(label,url)=>{const a=node('a',label,'detail-link');a.href=url;a.target='_blank';a.rel='noopener noreferrer';return a;};
 const section=(title)=>{const box=node('section','','detail-section');box.append(node('h3',title));return box;};
 function renderProduct(p){
  if(chosenSize===null && (p.sizes||'').split('|').filter(Boolean).length===1)chosenSize=p.sizes;
  content.replaceChildren();
  const layout=node('div','','detail-layout');
  const photoPanel=node('div');const image=node('img','','detail-image');image.alt=p.title||'Shirt';image.referrerPolicy='no-referrer';
  const missing=node('div','Image unavailable','detail-image-error');image.onerror=()=>{image.style.display='none';missing.style.display='grid';};
  photoPanel.append(image,missing);const gallery=node('div','','detail-gallery');photoPanel.append(gallery);
  const urls=[...new Set([p.image_url,...(p.listing_images||[]).filter(url=>!/[._](?:AC_)?(?:SR|US|SS)\d/i.test(url))].map(safeURL).filter(Boolean))].slice(0,12);
  const fullImage=link('Open full image',urls[0]||'#');
  function chooseImage(url,index){image.style.display='block';missing.style.display='none';image.src=url;fullImage.href=url;for(const [i,button] of [...gallery.children].entries())button.setAttribute('aria-pressed',String(i===index));}
  for(const [index,url] of urls.entries()){
   const button=node('button');button.type='button';button.setAttribute('aria-label','View listing image '+(index+1));button.setAttribute('aria-pressed','false');
   const thumbnail=node('img');thumbnail.src=url;thumbnail.alt='';thumbnail.loading='lazy';thumbnail.referrerPolicy='no-referrer';button.append(thumbnail);button.onclick=()=>chooseImage(url,index);gallery.append(button);
  }
  if(urls.length){chooseImage(urls[0],0);photoPanel.append(fullImage);}else{image.style.display='none';missing.style.display='grid';}
  const info=node('div');info.append(node('div',(p.brand_source==='store'?'Store: ':'Brand: ')+(p.brand||p.store||'Not recorded'),'detail-brand'));
  const title=node('h2',p.title||p.parent_asin,'detail-title');title.id='product-heading';info.append(title);
  info.append(node('p',`${p.average_rating??'—'} / 5 · ${(p.rating_number||0).toLocaleString()} ratings`,'rating'));
  info.append(node('div',p.price_inr_est==null?'Price unavailable':'₹'+Math.round(p.price_inr_est).toLocaleString('en-IN'),'detail-price'));
  info.append(node('p',(p.price_source||'').startsWith('exact')?'2023 listing price':'Estimated price','detail-note'));
  info.append(node('h3','Select size'));
  const sizeOptions=new Map();
  for(const variant of [p,...(p.variants||[])])for(const size of (variant.sizes||'').split('|').filter(Boolean)){
   if(!sizeOptions.has(size))sizeOptions.set(size,variant.parent_asin);
  }
  const sizes=[...sizeOptions.keys()];const sizeBox=node('div','','detail-sizes');
  for(const size of sizes){
   const button=node('button',size,'detail-size');button.type='button';
   button.setAttribute('aria-pressed',String(chosenSize===size));
   button.onclick=()=>window.openProduct(sizeOptions.get(size),size);sizeBox.append(button);
  }
  info.append(sizeBox);
  let witnesses=[];try{witnesses=JSON.parse(p.size_evidence||'[]');}catch{}
  if(!sizes.length)info.append(node('p',witnesses.length?'See size details below.':'Size not recorded.','detail-note'));
  if(sizes.length)info.append(node('p','Recorded sizes · current stock may vary','detail-note'));
  if(witnesses.length){const evidence=node('details');evidence.append(node('summary','Size details'));for(const witness of witnesses.slice(0,12))evidence.append(node('p',witness.text,'detail-note'));info.append(evidence);}
  const facts=node('dl','','detail-facts');
  const labels=[['Color',(p.color||'').split('|').filter(Boolean).join(', ')||'Not recorded'],
   ['Pattern',valueOrUnknown(p.pattern)],['Sleeves',p.sleeve==='full'?'Full sleeve':p.sleeve==='half'?'Half sleeve':'Not recorded'],
   ['Fabric',valueOrUnknown(p.fabric)],['Fit',valueOrUnknown(p.fit_type)],
   ['Pockets',({no:'No pockets',single:'Single pocket',double:'Double pockets'})[p.pocket]||'Not recorded'],
   ];
  for(const [label,value] of labels)facts.append(node('dt',label),node('dd',value));info.append(facts);
  if(p.vision_status!=='ok')info.append(node('p','Color and pattern are image-based estimates.','detail-note'));
  if(/^[A-Z0-9]{10}$/.test(p.parent_asin))info.append(link('Open Amazon listing','https://www.amazon.com/dp/'+p.parent_asin));
  layout.append(photoPanel,info);content.append(layout);
  if(p.features?.length){const about=section('About this shirt');const list=node('ul');for(const feature of p.features)list.append(node('li',feature));about.append(list);content.append(about);}
  if(p.description?.length){const description=section('Product description');for(const paragraph of p.description)description.append(node('p',paragraph));content.append(description);}
  const entries=Object.entries(p.listing_details||{});
  if(entries.length){const extra=node('details','','detail-section');extra.append(node('summary','More product information'));const table=node('table','','detail-listing');const body=node('tbody');for(const [key,value] of [['Product ID',p.parent_asin],...entries]){const row=node('tr');const heading=node('th',key);heading.scope='row';row.append(heading,node('td',typeof value==='object'?JSON.stringify(value):String(value)));body.append(row);}table.append(body);extra.append(table);content.append(extra);}
  content.append(node('p','Amazon 2023 catalog · prices and availability may have changed.','detail-note'));
 }
 window.openProduct=async (parentAsin,size=null)=>{
  if(!standalone){window.open('/product/'+encodeURIComponent(parentAsin),'_blank','noopener,noreferrer');return;}
  chosenSize=size;
  if(pending)pending.abort();const request=new AbortController();pending=request;
  const heading=node('h2','Loading shirt details…','detail-loading');heading.id='product-heading';content.replaceChildren(heading);
  window.scrollTo({top:0,behavior:'smooth'});
  try{
   const response=await fetch('/api/products/'+encodeURIComponent(parentAsin),{signal:request.signal});const data=await response.json();
   if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'Could not load this product.');
   if(pending===request){renderProduct(data);document.title=(data.title||'Shirt details')+' · SmartShirts';history.replaceState(null,'','/product/'+encodeURIComponent(parentAsin)+(size?'?size='+encodeURIComponent(size):''));}
  }catch(error){if(pending===request&&error.name!=='AbortError'){const heading=node('h2','Could not load shirt details');heading.id='product-heading';content.replaceChildren(heading,node('p',error.message,'error'));}}
 };
 if(standalone){const id=location.pathname.split('/').pop();window.openProduct(id,new URLSearchParams(location.search).get('size'));}
})();
