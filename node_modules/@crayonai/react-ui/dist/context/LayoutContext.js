import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext, useMemo } from "react";
export const LayoutContext = createContext({ layout: "fullscreen" });
export const LayoutContextProvider = ({ children, layout, }) => {
    const value = useMemo(() => ({ layout }), [layout]);
    return _jsx(LayoutContext.Provider, { value: value, children: children });
};
export const useLayoutContext = () => {
    const context = useContext(LayoutContext);
    return context;
};
//# sourceMappingURL=LayoutContext.js.map