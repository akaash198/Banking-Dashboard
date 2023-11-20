const toggleSidebar = () => document.body.classList.toggle("open");


let reviews = [
    {% for index, review in bank_reviews_df.iterrows %}
    {
        "review_id": "{{ review.reviewId }}",
        "user_name": "{{ review.userName }}",
        "user_image": "{{ review.userImage }}",
        "content": "{{ review.content }}",
        "date": "{{ review.at }}",
        "app_version": "{{ review.appVersion }}",
    },
    {% endfor %}

]

let tableRowCount = document.getElementsByClassName('table-row-count');
tableRowCount[0].innerHTML = `(${reviews.length}) reviews`;
console.log(tableRowCount)
   
let tableBody = document.getElementById('team-member-rows');
const itemsOnPage = 5; 
const numberOfPages = Math.ceil(reviews.length / itemsOnPage);

const start = (new URLSearchParams(window.location.search)).get('page') || 1;

const mappedRecords = reviews
.filter((_, index) => index >= (start - 1) * itemsOnPage && index < start * itemsOnPage)
.map((review) => {
    return `
        <tr>
            <td class="team-member-profile">
                <img src="${review.user_image}" alt="${review.user_name}">
                <span class="profile-info">
                    <span class="profile-info__name">${review.user_name}</span>
                </span>
            </td>
            <td class="team-member-review">${review.review_id}</td>
            <td class="team-member-content">${review.content}</td>
            <td class="team-member-date">${review.date}</td>
            <td class="team-member-app-version">${review.app_version}</td>


                
        </tr>
    `
});

tableBody.innerHTML = mappedRecords.join('');