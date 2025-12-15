import { useState } from "react";
export const useComposerState = () => {
    const [textContent, setTextContent] = useState("");
    return { textContent, setTextContent };
};
//# sourceMappingURL=useComposerState.js.map