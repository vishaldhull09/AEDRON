const input_photo = document.querySelector('#input_photo');
const input_video = document.querySelector('#input_video');

const image_drop = document.querySelector('#imgupload .drop');
const video_drop = document.querySelector('#vidupload .drop');


image_drop.addEventListener('dragenter', dragStart);
video_drop.addEventListener('dragenter', dragStart);


function dragStart(e) {
    if (e.target.parentElement.parentElement.id == 'imgupload') {
        image_drop.style.border = "4px dashed #00ADB5";
        image_drop.style.background = "rgba(0, 153, 255, .05)";
    } else {
        video_drop.style.border = "4px dashed #00ADB5";
        video_drop.style.background = "rgba(0, 153, 255, .05)";
    }
}

const events = ['dragleave', 'dragend', 'mouseout', 'drop'];
events.forEach(event => {
    image_drop.addEventListener(event, dragOver);
    video_drop.addEventListener(event, dragOver);
});

function dragOver(e) {
    if (e.target.parentElement.parentElement.id == 'imgupload') {
        image_drop.style.border = "3px dashed #adadad";
        image_drop.style.background = "transparent";
    } else {
        video_drop.style.border = "3px dashed #adadad";
        video_drop.style.background = "transparent";
    }
}

function handleImageFileSelect(evt) {
    const f = evt.target.files[0];
    const h1 = image_drop.querySelector('h1');
    if (!f.type.startsWith('image/')) {
        h1.style.display = 'block';
        h1.textContent = 'Please Select an Image';
        dropImage.style.opacity = 0;
        document.querySelector('#submitImg').classList.add('disabled');

    } else {
        const reader = new FileReader();
        reader.onload = (function() {
            return function(e) {
                const dropImage = document.querySelector('#dropImage');
                h1.style.display = 'none';
                dropImage.src = e.target.result;
                dropImage.style.opacity = 1;
                document.querySelector('#submitImg').classList.remove('disabled');
            };
        })(f);
        reader.readAsDataURL(f);
    }
}

function handleVideoFileSelect(evt) {
    const f = evt.target.files[0];
    const h1 = video_drop.querySelector('h1');
    if (!f.type.startsWith('video/')) {
        h1.textContent = 'Please Select a Video';
        document.querySelector('#submitVideo').classList.add('disabled');

    } else {
        h1.textContent = f.name;
        document.querySelector('#submitVideo').classList.remove('disabled');
    }
}

input_photo.onchange = handleImageFileSelect;
input_video.onchange = handleVideoFileSelect;