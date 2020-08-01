document.body.onload = () => {
    const imgResult = document.getElementById('imgResult');
    const videoResult = document.getElementById('videoResult');
    if (videoResult) {
        scrollTo(0, document.body.scrollHeight);
    } else if (imgResult) {
        imgResult.scrollIntoView();
    }
    const buttons = document.querySelectorAll(".tabs button");
    buttons.forEach(button => {
        if (button.classList.contains('active')) {
            const selectedTabContent = document.getElementById(button.dataset.tab);
            selectedTabContent.classList.add('active');
        }
    })

}

function openTab(e) {
    const buttons = document.querySelectorAll(".tabs button")
    buttons.forEach(button => {
        button.classList.remove('active')
    })
    e.target.classList.add('active');

    const tabContents = document.querySelectorAll(".tab-content");
    tabContents.forEach(tabContent => {
        tabContent.classList.remove('active');

    })
    const selectedTabContent = document.getElementById(e.target.dataset.tab);
    selectedTabContent.classList.add('active');
}

function ChangePhoto(e, name, img) {
    img = typeof img !== 'undefined' ? img : "{{ result['original'] }}";
    const index = e.target.dataset.index;

    document.querySelectorAll(".label")[index].innerHTML = name;
    document.querySelectorAll(".detectedImg>img")[index].src = img;
}

function Submit(upName) {
    document.getElementById(upName).submit();
}