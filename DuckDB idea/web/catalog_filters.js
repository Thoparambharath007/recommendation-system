/* All optional selections are explicit; no default brand or attribute is sent. */
(() => {
 const $=id=>document.getElementById(id);
 const make=(tag,value='',cls='')=>{const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e;};
 const fields=['color','pattern','size','sleeve','fabric','fit','pocket','max_price'];
 const selectedBrands=new Set();let brands=[],ready=false,applied={},searchTimer;
 const names={color:'Color',pattern:'Pattern',size:'Size',sleeve:'Sleeves',fabric:'Fabric',fit:'Fit',pocket:'Pockets',max_price:'Max price'};
 const labels={full:'Full sleeve',half:'Half sleeve',no:'No pockets',single:'Single pocket',double:'Double pockets'};
 const human=value=>labels[value]||String(value).replaceAll('_',' ');
 function selected(){const value={};if(selectedBrands.size)value.brands=[...selectedBrands];for(const field of fields){const raw=$('filter-'+field).value;if(raw!=='')value[field]=field==='max_price'?Number(raw):raw;}return value;}
 function update(){
  $('brand-button').textContent=selectedBrands.size===0?'All brands ▾':selectedBrands.size===1?[...selectedBrands][0]+' ▾':selectedBrands.size+' brands selected ▾';
  const chips=$('filter-chips');chips.replaceChildren();
  for(const brand of selectedBrands){const chip=make('button',brand+' ×','filter-chip');chip.type='button';chip.setAttribute('aria-label','Remove brand '+brand);chip.onclick=()=>{selectedBrands.delete(brand);update();};chips.append(chip);}
  for(const field of fields){const raw=$('filter-'+field).value;if(raw==='')continue;const value=field==='max_price'?'₹'+Number(raw).toLocaleString('en-IN'):human(raw);const chip=make('button',names[field]+': '+value+' ×','filter-chip');chip.type='button';chip.setAttribute('aria-label','Remove '+names[field]+' filter');chip.onclick=()=>{$('filter-'+field).value='';update();};chips.append(chip);}
  if(ready)$('filter-status').textContent=JSON.stringify(selected())===JSON.stringify(applied)?(Object.keys(applied).length?'Filters applied':'All styles'):'Apply your changes';
 }
 function showBrands(){
  const needle=$('brand-search').value.trim().toLocaleLowerCase();const shown=brands.filter(b=>b.name.toLocaleLowerCase().includes(needle));
  $('brand-count').textContent=shown.length.toLocaleString()+' of '+brands.length.toLocaleString()+' brands · '+selectedBrands.size+' selected';
  const list=$('brand-list');list.replaceChildren();const fragment=document.createDocumentFragment();
  for(const brand of shown){const label=make('label','','brand-option');const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=selectedBrands.has(brand.name);checkbox.onchange=()=>{
    if(checkbox.checked&&selectedBrands.size>=50){checkbox.checked=false;$('brand-count').textContent='Select up to 50 brands. Clear a selection to choose another.';return;}
    if(checkbox.checked)selectedBrands.add(brand.name);else selectedBrands.delete(brand.name);update();$('brand-count').textContent=shown.length.toLocaleString()+' of '+brands.length.toLocaleString()+' brands · '+selectedBrands.size+' selected';
   };label.append(checkbox,make('span',brand.name),make('small',brand.products.toLocaleString()));fragment.append(label);}
  if(!shown.length)fragment.append(make('p','No brands match this name. Try another spelling.','brand-empty'));list.append(fragment);
 }
 window.getSelectedFilters=selected;
 window.markFiltersApplied=value=>{applied=value||{};update();};
 $('brand-button').onclick=()=>{$('brand-search').value='';showBrands();$('brand-dialog').showModal();$('brand-search').focus();};
 $('brand-search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(showBrands,80);};
 $('brand-clear').onclick=()=>{selectedBrands.clear();showBrands();update();};
 $('brand-apply').onclick=()=>{$('brand-dialog').close();window.runShirtSearch?.();};
 $('filter-apply').onclick=()=>window.runShirtSearch?.();
 for(const field of fields)$('filter-'+field).addEventListener(field==='max_price'?'input':'change',update);
 $('filter-clear').onclick=()=>{selectedBrands.clear();for(const field of fields)$('filter-'+field).value='';update();};
 const dialog=$('brand-dialog');dialog.addEventListener('click',event=>{if(event.target===dialog){const box=dialog.getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)dialog.close();}});
 (async()=>{try{
  const response=await fetch('/api/filters');if(!response.ok)throw new Error('Filter choices could not be loaded. Refresh to retry.');const options=await response.json();brands=options.brands;
  for(const [field,key] of [['color','colors'],['pattern','patterns'],['size','sizes'],['sleeve','sleeves'],['fabric','fabrics'],['fit','fits'],['pocket','pockets']]){
   const select=$('filter-'+field);for(const value of options[key]){const option=make('option',human(value));option.value=value;select.append(option);}select.disabled=false;
  }
  ready=true;$('brand-button').disabled=false;update();
 }catch(error){$('brand-button').textContent='Brands unavailable';$('filter-status').textContent=error.message;}})();
})();
