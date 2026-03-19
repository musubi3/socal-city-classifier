const container = d3.select("#loss-chart-container");
const width = container.node().getBoundingClientRect().width;
const height = container.node().getBoundingClientRect().height;

const margin = { top: 30, right: 30, bottom: 50, left: 50 };

const svg = container.append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`)
    .style("max-width", "100%")
    .style("height", "auto");

d3.json("web/lib/losses.json").then(function (data) {

    const parsedData = data.map((d, i) => ({ iteration: i, loss: d }));

    const x = d3.scaleLinear()
        .domain(d3.extent(parsedData, d => d.iteration))
        .range([margin.left, width - margin.right]);

    const y = d3.scaleLinear()
        .domain([0, d3.max(parsedData, d => d.loss)])
        .nice()
        .range([height - margin.bottom, margin.top]);

    const area = d3.area()
        .x(d => x(d.iteration))
        .y0(y(0))
        .y1(d => y(d.loss))
        .curve(d3.curveMonotoneX);

    svg.append("path")
        .datum(parsedData)
        .attr("class", "chart-area")
        .attr("d", area);

    const line = d3.line()
        .x(d => x(d.iteration))
        .y(d => y(d.loss))
        .curve(d3.curveMonotoneX);

    svg.append("path")
        .datum(parsedData)
        .attr("class", "chart-line")
        .attr("fill", "none")
        .attr("stroke-width", 2.5)
        .attr("d", line);

    svg.append("g")
        .attr("transform", `translate(0,${height - margin.bottom})`)
        .attr("class", "chart-axis")
        .call(d3.axisBottom(x).ticks(5).tickSizeOuter(0))
        .call(g => g.append("text")
            .attr("x", width / 2)
            .attr("y", 35)
            .attr("fill", "currentColor")
            .attr("text-anchor", "middle")
            .style("font-size", "14px")
            .text("Iteration"));

    svg.append("g")
        .attr("transform", `translate(${margin.left},0)`)
        .attr("class", "chart-axis")
        .call(d3.axisLeft(y).ticks(5))
        .call(g => g.select(".domain").remove())
        .call(g => g.append("text")
            .attr("x", 0)
            .attr("y", margin.top - 10)
            .attr("fill", "currentColor")
            .attr("text-anchor", "start")
            .style("font-size", "14px")
            .text("↑ Empirical Risk (Loss)"));

    /** TOOLTIP */
    const tooltip = d3.select("#loss-chart-container")
        .append("div")
        .attr("class", "html-tooltip");

    const focusDot = svg.append("g")
        .attr("class", "focus")
        .style("display", "none");

    // focusDot.append("circle")
    //     .attr("r", 5)
    //     .attr("class", "tooltip-dot");

    svg.append("rect")
        .attr("class", "overlay")
        .attr("width", width)
        .attr("height", height)
        .style("fill", "none")
        .style("pointer-events", "all")
        .on("mouseover", () => {
            focusDot.style("display", null);
            tooltip.style("opacity", 1); 
        })
        .on("mouseout", () => {
            focusDot.style("display", "none");
            tooltip.style("opacity", 0);
        })
        .on("mousemove", mousemove);

    const bisectIteration = d3.bisector(d => d.iteration).left;

    function mousemove(event) {
        const x0 = x.invert(d3.pointer(event)[0]);
        const i = bisectIteration(parsedData, x0, 1);
        const d0 = parsedData[i - 1];
        const d1 = parsedData[i];
        if (!d0 || !d1) return;
        const d = x0 - d0.iteration > d1.iteration - x0 ? d1 : d0;

        focusDot.attr("transform", `translate(${x(d.iteration)},${y(d.loss)})`);

        tooltip.html(`Iter: ${d.iteration}<br>Loss: ${d.loss.toFixed(3)}`);

        const [mouseX, mouseY] = d3.pointer(event);

        if (mouseX > width - 120) {
            tooltip.style("left", (mouseX - 110) + "px")
                .style("top", (mouseY + 20) + "px");
        } else {
            tooltip.style("left", (mouseX + 20) + "px")
                .style("top", (mouseY + 20) + "px");
        }
    }
});