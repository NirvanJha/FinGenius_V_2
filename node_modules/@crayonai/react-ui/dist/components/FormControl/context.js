import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext } from "react";
const FormControlContext = createContext(null);
export function useFormControlContext() {
    return useContext(FormControlContext);
}
export function FormControlProvider(props) {
    const { value, children } = props;
    return _jsx(FormControlContext.Provider, { value: value, children: children });
}
//# sourceMappingURL=context.js.map