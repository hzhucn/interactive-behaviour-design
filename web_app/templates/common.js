function updateSelector(select, list) {
    for (var i = 0; i < list.length; i++) {
        if (i < select.childElementCount) {
            opt = select.childNodes[1 + i];
        } else {
            opt = document.createElement('option');
            select.appendChild(opt);
        }
        opt.value = list[i];
        opt.innerHTML = list[i];
    }
}

function updateLabelList() {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "get_label_list", false);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(null);
    labels = JSON.parse(xhr.responseText);

    var select = document.getElementById("label_select");
    updateSelector(select, labels);
    if (labels.length > 0) {
        select.selectedIndex = 0;
    }
}