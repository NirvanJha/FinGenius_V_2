import { jsx as _jsx } from "react/jsx-runtime";
import clsx from "clsx";
const variants = {
    clear: "text-content-clear",
    card: "text-content-card",
    sunk: "text-content-sunk",
};
const TextContent = (props) => {
    const { children, variant = "sunk", className, style } = props;
    return (_jsx("div", { className: clsx("text-content", variants[variant], className), style: style, children: children }));
};
export { TextContent };
//# sourceMappingURL=TextContent.js.map