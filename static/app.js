const fileInput=document.getElementById("fileInput");
const dropZone=document.getElementById("dropZone");
const uploadCard=document.getElementById("uploadCard");
const resultGrid=document.getElementById("resultGrid");
const previewImg=document.getElementById("previewImg");
const newBtn=document.getElementById("newBtn");
dropZone.addEventListener("dragover",e=>{e.preventDefault();dropZone.classList.add("drag-over")});
dropZone.addEventListener("dragleave",()=>dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop",e=>{e.preventDefault();dropZone.classList.remove("drag-over");const f=e.dataTransfer.files[0];if(f&&f.type.startsWith("image/"))handleFile(f)});
dropZone.addEventListener("click",()=>fileInput.click());
fileInput.addEventListener("change",()=>{if(fileInput.files[0])handleFile(fileInput.files[0])});
newBtn.addEventListener("click",()=>{uploadCard.style.display="block";resultGrid.style.display="none";fileInput.value=""});
function handleFile(file){
  const reader=new FileReader();
  reader.onload=e=>{previewImg.src=e.target.result};
  reader.readAsDataURL(file);
  uploadCard.style.display="none";
  resultGrid.style.display="grid";
  showLoading();
  const formData=new FormData();
  formData.append("file",file);
  fetch("/predict",{method:"POST",body:formData})
    .then(r=>r.json())
    .then(data=>{if(data.error)showError(data.error);else showResult(data)})
    .catch(err=>showError("Network error: "+err.message));
}
function showLoading(){
  document.getElementById("loadingState").style.display="flex";
  document.getElementById("resultState").style.display="none";
  document.getElementById("errorState").style.display="none";
}
function showError(msg){
  document.getElementById("loadingState").style.display="none";
  document.getElementById("resultState").style.display="none";
  document.getElementById("errorState").style.display="block";
  document.getElementById("errorMsg").textContent="❌ "+msg;
}
function showResult(data){
  document.getElementById("loadingState").style.display="none";
  document.getElementById("errorState").style.display="none";
  document.getElementById("resultState").style.display="block";
  const isMalignant=data.predicted_class==="Malignant";
  const verdictBox=document.getElementById("verdictBox");
  verdictBox.className="verdict "+(isMalignant?"malignant":"benign");
  document.getElementById("verdictValue").textContent=isMalignant?"⚠ Malignant":"✓ Benign";
  setTimeout(()=>{
    document.getElementById("benignBar").style.width=data.benign_pct+"%";
    document.getElementById("malignantBar").style.width=data.malignant_pct+"%";
  },100);
  document.getElementById("benignPct").textContent=data.benign_pct+"%";
  document.getElementById("malignantPct").textContent=data.malignant_pct+"%";
  document.getElementById("confidenceVal").textContent=data.confidence+"%";
}