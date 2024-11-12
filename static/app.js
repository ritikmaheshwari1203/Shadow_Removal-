console.log("hello")

const nav = document.querySelector("nav");
const header = document.querySelector("header");
const listItems = document.querySelectorAll('ul li');

const options = {
  threshold: 0.27 // Trigger the function when 80% of the section is visible
};

function handleIntersection(entries, observer) {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      nav.classList.toggle('bg-gray-200', false);
      nav.classList.toggle('text-white', true);
      listItems.forEach(item => item.classList.toggle('text-white', true));
    } else {
      nav.classList.toggle('bg-gray-200', true);
      nav.classList.toggle('text-white', false);
      listItems.forEach(item => item.classList.toggle('text-white', false));
    }
  });
}

const observer = new IntersectionObserver(handleIntersection, options);

observer.observe(header);

const gallery=document.querySelectorAll(".gallery .image");
// const image=document.querySelectorAll(".gallery .image src");

const preview=document.querySelector(".preview-box");
selectimg=preview .querySelector("img");
window.onload=()=>{

    for (let index = 0; index < gallery.length; index++) {
        // console.log(index);
        gallery[index].onclick =()=>{
            let newindex=index;
            console.log(newindex);
            preview.classList.remove("hidden");
            function preview1(){
                let selectedimgurl=gallery[newindex].querySelector("img").src;
                // console.log(selectedimgurl);
                selectimg.src=selectedimgurl;
                // selectimg=selectedimgurl
            }
            const prevBtn= document.querySelector(".prev")
            const nexBtn=document.querySelector(".next")
            if (newindex == 0) {
                prevBtn.style.display="none";
            }
            else{
                prevBtn.style.display="block"
            }
            if (newindex >=gallery.length-1) {
                nexBtn.style.display="none";
            }
            else{
                nexBtn.style.display="block";
            }
            prevBtn.onclick=()=>{
                newindex--;
                if (newindex==0) {
                    preview1();
                    prevBtn.style.display="none";
                }
                else{

                    preview1();
                }
            }
            nexBtn.onclick=()=>{
                newindex++;
                if (newindex >=gallery.length-1) {
                    preview1();
                    nexBtn.style.display="none";
                }
                else{

                    preview1();
                }
            }
            preview1();


        }
        
const close=document.getElementById("close");
close.onclick=()=>{

    preview.classList.add("hidden")
}
        
    }
}


function change(e){
    let list=document.querySelector('ul');
    if(e.name==='menu'){
        e.name="close";
        list.classList.remove('top-[-170px]');
        list.classList.add('top-[6.1rem]');
        // list.classList.add('.transion');
        list.classList.add('left-[-1px]');
        list.classList.add('w-full');
        list.classList.add('bg-white');
        list.classList.add('text-center');
        list.classList.add('pb-3');
        list.classList.add('pt-5');
    }
    else if(e.name==='close'){
        e.name="menu"
        list.classList.remove('top-[6.1rem]');
        list.classList.add('top-[-170px]');
        // list.classList.remove('w-full');
        // list.classList.remove('transition-all ease-in duration-500');
    }
}